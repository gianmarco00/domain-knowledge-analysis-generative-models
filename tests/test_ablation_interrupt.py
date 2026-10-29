import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.name != "posix", reason="The test sends SIGINT to a POSIX process group.")
def test_ctrl_c_stops_spawned_ablation_workers(tmp_path):
    script = tmp_path / "interrupt_ablation.py"
    script.write_text('''
import os
from pathlib import Path
import sys
import time

from domain_knowledge_analysis.experiments import ablation as ablation_module


def slow_training(config_path):
    Path(config_path).parent.joinpath("worker.pid").write_text(str(os.getpid()))
    time.sleep(60)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    ablation_module.train_ablation_model = slow_training
    study = ablation_module.Ablation.__new__(ablation_module.Ablation)
    study.ablation_config = {"parallel_models": 2}
    study.runs = []
    for name in ("one", "two"):
        model_dir = root / name
        model_dir.mkdir()
        study.runs.append({"model_dir": model_dir, "config_path": model_dir / "config.yaml"})
    study.prepare_dataset_caches = lambda runs: None
    try:
        study.train_models()
    except KeyboardInterrupt:
        sys.exit(130)
''')
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root / "src")
    environment.setdefault("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    process = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)],
        cwd=repo_root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    worker_pids = []
    try:
        deadline = time.monotonic() + 30
        pid_files = [tmp_path / name / "worker.pid" for name in ("one", "two")]
        while time.monotonic() < deadline and not all(path.exists() for path in pid_files):
            if process.poll() is not None:
                pytest.fail("The ablation process exited before both workers started.")
            time.sleep(0.05)
        assert all(path.exists() for path in pid_files), "Workers did not start in time."
        worker_pids = [int(path.read_text()) for path in pid_files]

        os.killpg(process.pid, signal.SIGINT)
        assert process.wait(timeout=10) == 130
        for pid in worker_pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        for pid in worker_pids:
            try:
                if os.getpgid(pid) == process.pid:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
