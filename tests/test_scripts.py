"""The board scripts: install.sh (fresh install, refusal, STUDIO_UPDATE), setup.sh and webapp/run.sh.

Nothing here needs a board: install.sh runs from a local STUDIO_ARCHIVE built from the
working tree, and setup.sh runs against a stand-in pyneat venv with HOME in a tmp dir.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["install.sh", "setup.sh", "run.sh", "deploy.sh", "webapp/run.sh",
           "webapp/make_cert.sh", "webapp/desktop-launch.sh", "webapp/models_dir.sh"]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def run(cmd, env=None, cwd=None, timeout=120):
    full = {k: v for k, v in os.environ.items() if not k.startswith(("STUDIO_", "CYWS_", "CREATE_"))}
    full["NO_COLOR"] = "1"
    full.update(env or {})
    # start_new_session: no controlling terminal, so setup.sh never prompts.
    return subprocess.run(cmd, env=full, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, stdin=subprocess.DEVNULL, start_new_session=True)


def make_archive(path: Path, top: str = "neat-change-studio", files=None, extra: dict | None = None) -> Path:
    """A .tar.gz shaped like GitHub's: one top-level directory holding the repo."""
    files = files if files is not None else [
        "install.sh", "setup.sh", "run.sh", "README.md", "requirements.txt",
        "webapp/run.sh", "webapp/make_cert.sh", "webapp/desktop-launch.sh", "webapp/models_dir.sh", "webapp/server.py",
        "data/testcyws_manifest.example.tsv"]
    with tarfile.open(path, "w:gz") as tar:
        for f in files:
            tar.add(ROOT / f, arcname=f"{top}/{f}")
        for name, text in (extra or {}).items():
            src = path.parent / ("extra_" + name.replace("/", "_"))
            src.write_text(text)
            tar.add(src, arcname=f"{top}/{name}")
    return path


@pytest.mark.parametrize("script", SCRIPTS)
def test_bash_syntax(script):
    r = run([BASH, "-n", str(ROOT / script)])
    assert r.returncode == 0, r.stderr


def test_install_fresh(tmp_path):
    arc = make_archive(tmp_path / "src.tar.gz")
    dest = tmp_path / "ncs"
    r = run([BASH, str(ROOT / "install.sh")], env={"STUDIO_ARCHIVE": str(arc), "STUDIO_DIR": str(dest)})
    assert r.returncode == 0, r.stderr
    assert "Installed in" in r.stdout and "./setup.sh" in r.stdout
    for f in ("run.sh", "setup.sh", "webapp/run.sh", "webapp/desktop-launch.sh"):
        assert (dest / f).is_file()
        assert os.access(dest / f, os.X_OK), f


def test_install_dir_as_argument(tmp_path):
    arc = make_archive(tmp_path / "src.tar.gz")
    r = run([BASH, str(ROOT / "install.sh"), str(tmp_path / "here")], env={"STUDIO_ARCHIVE": str(arc)})
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "here/webapp/run.sh").is_file()


def test_install_into_empty_dir(tmp_path):
    arc = make_archive(tmp_path / "src.tar.gz")
    (tmp_path / "empty").mkdir()
    r = run([BASH, str(ROOT / "install.sh")], env={"STUDIO_ARCHIVE": str(arc), "STUDIO_DIR": str(tmp_path / "empty")})
    assert r.returncode == 0, r.stderr


def test_install_refuses_non_empty_dir(tmp_path):
    arc = make_archive(tmp_path / "src.tar.gz")
    dest = tmp_path / "busy"
    dest.mkdir()
    (dest / "notes.txt").write_text("mine")
    r = run([BASH, str(ROOT / "install.sh")], env={"STUDIO_ARCHIVE": str(arc), "STUDIO_DIR": str(dest)})
    assert r.returncode != 0
    assert "not empty" in r.stderr
    assert sorted(p.name for p in dest.iterdir()) == ["notes.txt"]


def test_install_refuses_existing_install_and_says_how_to_update(tmp_path):
    arc = make_archive(tmp_path / "src.tar.gz")
    dest = tmp_path / "ncs"
    env = {"STUDIO_ARCHIVE": str(arc), "STUDIO_DIR": str(dest)}
    assert run([BASH, str(ROOT / "install.sh")], env=env).returncode == 0
    r = run([BASH, str(ROOT / "install.sh")], env=env)
    assert r.returncode != 0
    assert "webapp/run.sh update" in r.stderr and "neat-change update" in r.stderr
    # Older installs (and deploy.sh's) have no `update` in webapp/run.sh: the curl form works on any.
    assert "STUDIO_UPDATE=1" in r.stderr and "install.sh | " in r.stderr


def test_install_rejects_other_archive(tmp_path):
    arc = make_archive(tmp_path / "other.tar.gz", files=["README.md", "run.sh"])
    r = run([BASH, str(ROOT / "install.sh")], env={"STUDIO_ARCHIVE": str(arc), "STUDIO_DIR": str(tmp_path / "x")})
    assert r.returncode != 0
    assert "not Neat Change Studio" in r.stderr
    assert not (tmp_path / "x").exists()


def test_install_rejects_unreadable_archive(tmp_path):
    bad = tmp_path / "bad.tar.gz"
    bad.write_text("not an archive")
    r = run([BASH, str(ROOT / "install.sh")], env={"STUDIO_ARCHIVE": str(bad), "STUDIO_DIR": str(tmp_path / "x")})
    assert r.returncode != 0
    assert "not a readable" in r.stderr


@pytest.mark.parametrize("no_rsync", [False, True])
def test_install_update_keeps_models_and_local_files(tmp_path, no_rsync):
    dest = tmp_path / "ncs"
    old = make_archive(tmp_path / "old.tar.gz", extra={"dropped.txt": "old file"})
    assert run([BASH, str(ROOT / "install.sh")], env={"STUDIO_ARCHIVE": str(old), "STUDIO_DIR": str(dest)}).returncode == 0
    kept = {"models/cyws_w8a16_mpk.tar.gz": "model", "data/testCYWS/images/a_ref.jpg": "pair",
            "out/one/a.json": "{}", "webapp/certs/key.pem": "key", "webapp/uploads/u.jpg": "u",
            "webapp/cache/c.npz": "c", "webapp/server.log": "log", ".venv/pyvenv.cfg": "venv",
            "build/onnx/cyws_fused.onnx": "onnx", "third_party/cyws/x.py": "tp", "venv/bin/python": "v",
            "eval/board_bf16/summary.csv": "eval", "weights.ckpt": "ckpt", ".env": "SECRET=1",
            "app/local.pem": "pem"}
    for name, text in kept.items():
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        (dest / name).write_text(text)
    (dest / "README.md").write_text("stale")

    new = make_archive(tmp_path / "new.tar.gz", extra={"NEW.txt": "new file"})
    env = {"STUDIO_ARCHIVE": str(new), "STUDIO_DIR": str(dest), "STUDIO_UPDATE": "1"}
    if no_rsync:   # PATH without rsync: the cp -a fallback
        bindir = tmp_path / "bin"
        bindir.mkdir()
        for tool in ("tar", "gzip", "grep", "mkdir", "mktemp", "cp", "rm", "chmod", "ls", "cat"):
            if shutil.which(tool):
                (bindir / tool).symlink_to(shutil.which(tool))
        env["PATH"] = str(bindir)
    r = run([BASH, str(ROOT / "install.sh")], env=env)
    assert r.returncode == 0, r.stderr
    assert "Updated" in r.stdout and "Updating" in r.stdout and "Installing" not in r.stdout
    for name, text in kept.items():
        assert (dest / name).read_text() == text, name
    assert (dest / "README.md").read_text() == (ROOT / "README.md").read_text()
    assert (dest / "NEW.txt").is_file()
    assert (dest / "data/testcyws_manifest.example.tsv").is_file()
    # rsync --delete removes what the new source dropped; the cp fallback leaves it.
    assert (dest / "dropped.txt").exists() == no_rsync


def test_install_update_with_nothing_there_is_a_fresh_install(tmp_path):
    arc = make_archive(tmp_path / "src.tar.gz")
    dest = tmp_path / "ncs"
    r = run([BASH, str(ROOT / "install.sh")], env={"STUDIO_ARCHIVE": str(arc), "STUDIO_DIR": str(dest), "STUDIO_UPDATE": "1"})
    assert r.returncode == 0, r.stderr
    assert "Installed in" in r.stdout and (dest / "setup.sh").is_file()


def test_setup_help_lists_options_and_environment():
    r = run([BASH, str(ROOT / "setup.sh"), "--help"])
    assert r.returncode == 0
    for word in ("--check-only", "--check", "--fetch", "--install", "--no-prompt",
                 "PYNEAT_VENV", "CYWS_MODELS", "CREATE_ALIAS", "CREATE_DESKTOP_ICON", "NO_COLOR"):
        assert word in r.stdout, word


def test_setup_rejects_unknown_option():
    r = run([BASH, str(ROOT / "setup.sh"), "--frobnicate"])
    assert r.returncode == 2


def test_setup_without_venv_fails(tmp_path):
    r = run([BASH, str(ROOT / "setup.sh")], env={"PYNEAT_VENV": str(tmp_path / "none"), "HOME": str(tmp_path)})
    assert r.returncode == 1
    assert "pyneat environment not found" in r.stderr


@pytest.fixture()
def board(tmp_path):
    """An install from the working tree, a stand-in pyneat venv (this Python plus a fake
    pyneat module) and a HOME with a Desktop, all in tmp_path."""
    pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    pytest.importorskip("flask")
    arc = make_archive(tmp_path / "src.tar.gz")
    app = tmp_path / "ncs"
    assert run([BASH, str(ROOT / "install.sh")], env={"STUDIO_ARCHIVE": str(arc), "STUDIO_DIR": str(app)}).returncode == 0
    venv = tmp_path / "pyneat"
    (venv / "bin").mkdir(parents=True)
    (venv / "lib").mkdir()
    (venv / "lib/pyneat.py").write_text("def build_info():\n    return 'stand-in 0.0'\n")
    (venv / "bin/python3").symlink_to(sys.executable)
    (venv / "bin/activate").write_text(f'export PATH="{venv}/bin:$PATH"\nexport PYTHONPATH="{venv}/lib"\n')
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    (home / ".bash_profile").write_text("# mine\n")
    env = {"PYNEAT_VENV": str(venv), "HOME": str(home), "XDG_DATA_HOME": str(home / ".local/share"),
           "XDG_DESKTOP_DIR": str(home / "Desktop"), "DBUS_SESSION_BUS_ADDRESS": ""}
    return app, home, env


def test_setup_without_terminal_asks_nothing_and_changes_no_profile(board):
    app, home, env = board
    r = run([str(app / "setup.sh")], env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no model downloaded" in r.stdout
    assert "Setup complete" in r.stdout
    assert (home / ".bash_profile").read_text() == "# mine\n"
    assert list((home / "Desktop").iterdir()) == []
    assert not list(app.glob("models/*"))
    if shutil.which("openssl"):
        assert (app / "webapp/certs/cert.pem").is_file() and (app / "webapp/certs/key.pem").is_file()


def test_setup_deploy_call_shape_and_old_flags(board):
    """deploy.sh runs '$REMOTE/setup.sh' over ssh with no terminal; --install still works."""
    app, _, env = board
    assert run([str(app / "setup.sh"), "--install"], env=env).returncode == 0
    assert run([str(app / "setup.sh"), "--check-only"], env=env).returncode == 0


def test_setup_alias_and_icon(board):
    app, home, env = board
    env = dict(env, CREATE_ALIAS="1", CREATE_DESKTOP_ICON="1")
    r = run([str(app / "setup.sh")], env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    line = f"alias neat-change='{app}/webapp/run.sh'"
    assert (home / ".bash_profile").read_text().count(line) == 1
    icon = home / "Desktop/neat-change-studio.desktop"
    assert f"Path={app}" in icon.read_text()
    assert "desktop-launch.sh" in icon.read_text()
    assert (home / ".local/share/applications/neat-change-studio.desktop").is_file()
    assert "neat-change update" in r.stdout
    # Again, without the variables: nothing is added twice; this install's icon is kept up to date.
    r = run([str(app / "setup.sh")], env={k: v for k, v in env.items() if not k.startswith("CREATE_")})
    assert r.returncode == 0
    assert (home / ".bash_profile").read_text().count(line) == 1
    assert "already set" in r.stdout


def test_setup_unknown_model_is_a_failure(board):
    app, _, env = board
    r = run([str(app / "setup.sh")], env=dict(env, CYWS_MODELS="fp64"))
    assert r.returncode == 1
    assert "Unknown model" in r.stderr


def test_setup_check_only_reports_missing_packages(board, tmp_path):
    app, _, env = board
    venv = Path(env["PYNEAT_VENV"])
    (venv / "lib/flask.py").write_text("raise ImportError('hidden for the test')\n")
    r = run([str(app / "setup.sh"), "--check-only"], env=env)
    assert r.returncode == 1
    assert "flask" in r.stderr


def _update_stage(tmp_path, local_installer=True):
    app = tmp_path / "ncs"
    (app / "webapp").mkdir(parents=True)
    shutil.copy(ROOT / "webapp/run.sh", app / "webapp/run.sh")
    log = tmp_path / "calls"
    if local_installer:
        (app / "install.sh").write_text(f'# STUDIO_UPDATE\necho "local-install $STUDIO_UPDATE $STUDIO_DIR" >> {log}\n')
    (app / "setup.sh").write_text(f'#!/usr/bin/env bash\necho "setup $*" >> {log}\n')
    (app / "setup.sh").chmod(0o755)
    return app, log


def _fake_curl(tmp_path, log, ok=True):
    """A curl on PATH that 'downloads' a stand-in install.sh (or fails, as offline)."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    body = (f'echo "curl $*" >> {log}\n'
            'while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done\n')
    if ok:
        body += f'printf \'# STUDIO_UPDATE\\necho "remote-install $STUDIO_UPDATE $STUDIO_DIR" >> {log}\\n\' > "$out"\n'
    else:
        body += "exit 6\n"
    (bindir / "curl").write_text("#!/bin/sh\n" + body)
    (bindir / "curl").chmod(0o755)
    return {"PATH": f"{bindir}:{os.environ['PATH']}"}


def test_run_sh_update_with_archive_uses_local_install_then_setup(tmp_path):
    """webapp/run.sh update with STUDIO_ARCHIVE (offline): the install's own install.sh with
    STUDIO_UPDATE=1 over this install, then setup.sh --no-prompt."""
    app, log = _update_stage(tmp_path)
    r = run([BASH, str(app / "webapp/run.sh"), "update"], env={"STUDIO_ARCHIVE": str(tmp_path / "x.tar.gz")})
    assert r.returncode == 0, r.stderr
    assert log.read_text().splitlines() == [f"local-install 1 {app}", "setup --no-prompt"]


def test_run_sh_update_runs_the_latest_install_sh(tmp_path):
    """Without STUDIO_ARCHIVE, the branch's install.sh is downloaded and run, so a fix to it
    applies at once, and an install without install.sh (made by deploy.sh) can update."""
    app, log = _update_stage(tmp_path, local_installer=False)
    r = run([BASH, str(app / "webapp/run.sh"), "update"], env=_fake_curl(tmp_path, log))
    assert r.returncode == 0, r.stderr
    lines = log.read_text().splitlines()
    assert "raw.githubusercontent.com/dotimothy/neat-change-studio/main/install.sh" in lines[0]
    assert lines[1:] == [f"remote-install 1 {app}", "setup --no-prompt"]


def test_run_sh_update_offline_falls_back_to_local_install_sh(tmp_path):
    app, log = _update_stage(tmp_path)
    r = run([BASH, str(app / "webapp/run.sh"), "update"], env=_fake_curl(tmp_path, log, ok=False))
    assert r.returncode == 0, r.stderr
    assert log.read_text().splitlines()[1:] == [f"local-install 1 {app}", "setup --no-prompt"]


def test_run_sh_update_offline_without_install_sh_fails(tmp_path):
    app, log = _update_stage(tmp_path, local_installer=False)
    r = run([BASH, str(app / "webapp/run.sh"), "update"], env=_fake_curl(tmp_path, log, ok=False))
    assert r.returncode == 1
    assert "Could not download install.sh" in r.stderr
    assert "setup" not in log.read_text()


def test_run_sh_port_needs_a_value():
    r = run([BASH, str(ROOT / "webapp/run.sh"), "--port"])
    assert r.returncode == 2 and "--port needs a value" in r.stderr


def test_mla_busy_pattern_is_shared():
    import re
    pat = re.compile(r'MLA_BUSY_PATTERN:-([^}]*)\}')
    setup = pat.findall((ROOT / "setup.sh").read_text())
    runsh = pat.findall((ROOT / "webapp/run.sh").read_text())
    assert len(setup) == 1 and setup == runsh
    assert "[d]ofa" in setup[0] and "[w]ebapp/server.py" in setup[0]


def test_install_sh_reads_nothing_after_main():
    assert (ROOT / "install.sh").read_text().rstrip().splitlines()[-1] == 'main "$@"; exit $?'


def test_setup_alias_and_icon_with_special_characters(tmp_path):
    """A path with spaces, |, & and ' still gives a working alias (updated in place) and icon."""
    pytest.importorskip("numpy"); pytest.importorskip("cv2"); pytest.importorskip("PIL"); pytest.importorskip("flask")
    arc = make_archive(tmp_path / "src.tar.gz")
    app = tmp_path / "it's a|b&c"
    assert run([BASH, str(ROOT / "install.sh")], env={"STUDIO_ARCHIVE": str(arc), "STUDIO_DIR": str(app)}).returncode == 0
    venv = tmp_path / "pyneat"
    (venv / "bin").mkdir(parents=True)
    (venv / "lib").mkdir()
    (venv / "lib/pyneat.py").write_text("def build_info():\n    return 'stand-in 0.0'\n")
    (venv / "bin/python3").symlink_to(sys.executable)
    (venv / "bin/activate").write_text(f'export PATH="{venv}/bin:$PATH"\nexport PYTHONPATH="{venv}/lib"\n')
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    (home / ".bashrc").write_text("# mine\nalias neat-change='/old/place/webapp/run.sh'\n")
    env = {"PYNEAT_VENV": str(venv), "HOME": str(home), "XDG_DATA_HOME": str(home / ".local/share"),
           "XDG_DESKTOP_DIR": str(home / "Desktop"), "DBUS_SESSION_BUS_ADDRESS": "", "SHELL": "/bin/bash",
           "CREATE_ALIAS": "1", "CREATE_DESKTOP_ICON": "1"}
    r = run([str(app / "setup.sh")], env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    rc = (home / ".bashrc").read_text()
    assert "/old/place" not in rc and rc.startswith("# mine\n")
    # The alias, as bash reads it, names this install's run.sh.
    out = run([BASH, "-c", 'source "$1"; printf %s "${BASH_ALIASES[neat-change]}"', "x", str(home / ".bashrc")])
    assert out.stdout == str(app / "webapp/run.sh"), out.stdout + out.stderr
    entry = (home / "Desktop/neat-change-studio.desktop").read_text()
    assert "Terminal=true" in entry and "x-terminal-emulator" not in entry
    assert f'Exec="{app}/webapp/desktop-launch.sh"' in entry



def test_models_dir_rule(tmp_path):
    """webapp/models_dir.sh follows the same order as server.default_models_dir()."""
    helper = ROOT / "webapp" / "models_dir.sh"
    def resolve(env=None):
        r = run([BASH, "-c", f'source "{helper}"; cyws_models_dir "{tmp_path}"'], env=env)
        assert r.returncode == 0, r.stderr
        return r.stdout.strip()
    assert resolve({"CYWS_MODELS_DIR": str(tmp_path / "env")}) == str(tmp_path / "env")
    (tmp_path / ".models-dir").write_text(str(tmp_path / "kept") + "\n")
    assert resolve() == str(tmp_path / "kept")
    (tmp_path / ".models-dir").unlink()
    expect = "/media/nvme/models/cyws" if Path("/media/nvme").is_dir() else str(tmp_path / "models")
    assert resolve() == expect


def test_setup_remembers_models_dir_and_names_old_models(board, tmp_path):
    app, home, env = board
    (app / "models").mkdir(exist_ok=True)
    (app / "models" / "cyws_bf16_mpk.tar.gz").write_bytes(b"old")
    target = tmp_path / "shared models"
    r = run([str(app / "setup.sh"), "--no-prompt", "--models-dir", str(target)], env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (app / ".models-dir").read_text().strip() == str(target.resolve())
    assert target.is_dir()
    out = r.stdout + r.stderr
    assert "not used any more" in out and "mv " in out
    r = run([str(app / "setup.sh"), "--no-prompt", "--models-dir", "default"], env=env)
    assert r.returncode == 0 and not (app / ".models-dir").exists()
