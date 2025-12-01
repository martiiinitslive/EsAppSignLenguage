import sys
from pathlib import Path
import subprocess
import shutil

import pytest

import launcher
from launcher import find_venv_python, run_backend, run_frontend


def test_find_venv_python_uses_sys(tmp_path):
    root = tmp_path
    result = find_venv_python(root)
    assert result.endswith('python.exe') or result == sys.executable


def test_find_venv_python_prefers_venv(tmp_path):
    root = tmp_path
    venv_py = root / ".venv" / "Scripts"
    venv_py.mkdir(parents=True)
    pyfile = venv_py / "python.exe"
    pyfile.write_text("")  # create file
    assert find_venv_python(root) == str(pyfile)


def test_run_backend_missing_script(tmp_path):
    root = tmp_path
    with pytest.raises(FileNotFoundError):
        run_backend(root, sys.executable)


def test_run_backend_starts_process(monkeypatch, tmp_path):
    backend_dir = tmp_path / "app-back"
    backend_dir.mkdir()
    script = backend_dir / "run_api.py"
    script.write_text("print('ok')")

    called = {}

    class DummyProc:
        def __init__(self):
            self.pid = 4321

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        called['cmd'] = cmd
        called['kwargs'] = kwargs
        return DummyProc()

    monkeypatch.setattr(launcher.subprocess, 'Popen', fake_popen)

    proc = run_backend(tmp_path, sys.executable, new_console=False)
    assert hasattr(proc, 'pid')
    assert called['cmd'][0] == sys.executable
    assert str(script) in called['cmd'][1]


def test_run_frontend_missing_npm(monkeypatch, tmp_path):
    root = tmp_path
    front = root / 'app-front'
    front.mkdir()

    # Ensure launcher.shutil.which returns None for npm
    monkeypatch.setattr(launcher.shutil, 'which', lambda x: None)

    with pytest.raises(RuntimeError):
        run_frontend(root)


def test_run_frontend_installs_and_starts(monkeypatch, tmp_path):
    root = tmp_path
    front = root / "app-front"
    front.mkdir()

    events = {'check_call': False, 'popen': False, 'popen_cmd': None, 'popen_cwd': None}

    # Ensure launcher finds npm
    monkeypatch.setattr(launcher.shutil, 'which', lambda x: "npm")

    def fake_check_call(cmd, cwd=None):
        events['check_call'] = True
        events['check_call_cmd'] = cmd
        events['check_call_cwd'] = cwd
        return 0

    class DummyProc:
        def __init__(self, cmd, cwd):
            self.pid = 1111

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        events['popen'] = True
        events['popen_cmd'] = cmd
        events['popen_cwd'] = kwargs.get('cwd')
        return DummyProc(cmd, kwargs.get('cwd'))

    monkeypatch.setattr(launcher.subprocess, 'check_call', fake_check_call)
    monkeypatch.setattr(launcher.subprocess, 'Popen', fake_popen)

    proc = run_frontend(root, new_console=False)
    assert hasattr(proc, 'pid')
    assert events['check_call'] is True
    assert events['check_call_cmd'][0] == "npm"
    assert events['popen'] is True
    assert events['popen_cmd'][0] == "npm"
    assert events['popen_cmd'][1] == "start"


def test_run_frontend_uses_existing_node_modules(monkeypatch, tmp_path):
    root = tmp_path
    front = root / "app-front"
    front.mkdir()
    (front / "node_modules").mkdir()

    events = {'check_call_called': False, 'popen_called': False}

    monkeypatch.setattr(launcher.shutil, 'which', lambda x: "npm")

    def fake_check_call(cmd, cwd=None):
        events['check_call_called'] = True
        raise AssertionError("npm install should not be called when node_modules exists")

    class DummyProc:
        def __init__(self, cmd, cwd):
            self.pid = 2222

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        events['popen_called'] = True
        events['popen_cmd'] = cmd
        events['popen_cwd'] = kwargs.get('cwd')
        return DummyProc(cmd, kwargs.get('cwd'))

    monkeypatch.setattr(launcher.subprocess, 'check_call', fake_check_call)
    monkeypatch.setattr(launcher.subprocess, 'Popen', fake_popen)

    proc = run_frontend(root, new_console=False)
    assert hasattr(proc, 'pid')
    assert events['popen_called'] is True
    assert events['popen_cmd'][0] == "npm"
    assert events['popen_cmd'][1] == "start"

