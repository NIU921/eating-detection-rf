"""Windows/PyCharm入口：先smoke，再把DEFAULT_STAGE改为all运行完整实验。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


# 在PyCharm直接右键运行本文件时，先使用smoke检查一个真实ZIP。
# 确认能完成后，将下一行的"smoke"改为"all"，再次运行。
DEFAULT_STAGE ="all"
DEFAULT_PREP_WORKERS = 1
DEFAULT_TRAIN_JOBS = 4
ROOT = Path(__file__).resolve().parent


def run_step(arguments, log_path):
    environment=os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1",PYTHONIOENCODING="utf-8",
                       OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    command=[sys.executable,"-u","-m",*arguments]
    print("RUN",subprocess.list2cmdline(command),flush=True)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace")
        try:
            for line in process.stdout:
                print(line,end="",flush=True); log.write(line); log.flush()
            return_code=process.wait()
        except KeyboardInterrupt:
            process.terminate(); process.wait()
            raise
    if return_code:
        raise SystemExit(f"步骤失败，后续步骤未执行。日志: {log_path}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("smoke","check","prepare","train","all"),default=DEFAULT_STAGE)
    parser.add_argument("--imu-dir",type=Path,default=ROOT/"IMU+PPG")
    parser.add_argument("--source-mode",choices=("auto","directory","zip"),default="auto")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--workers",type=int,default=DEFAULT_PREP_WORKERS)
    parser.add_argument("--jobs",type=int,default=DEFAULT_TRAIN_JOBS)
    parser.add_argument("--signal-config",type=Path,default=ROOT/"experiments"/"signal_baseline.json")
    parser.add_argument("--train-config",type=Path,default=ROOT/"experiments"/"train_baseline.json")
    args=parser.parse_args()
    output=args.output or ROOT/"runs"/("local_smoke" if args.stage=="smoke" else "local_baseline")
    output=output.resolve(); imu_dir=args.imu_dir.resolve()
    print(f"Python: {sys.executable}\nProject: {ROOT}\nStage: {args.stage}\nOutput: {output}",flush=True)
    if args.stage in ("check","all"):
        run_step(["dominant_hand.check_data","--root",str(ROOT),"--imu-dir",str(imu_dir),
                  "--source-mode",args.source_mode,
                  "--output",str(output/"preflight.json")],output/"logs"/"01_check.log")
    if args.stage in ("smoke","prepare","all"):
        arguments=["dominant_hand.prepare","--root",str(ROOT),"--imu-dir",str(imu_dir),
                   "--source-mode",args.source_mode,
                   "--config",str(args.signal_config.resolve()),"--workers",str(args.workers),
                   "--output",str(output/"dataset")]
        if args.stage=="smoke": arguments.extend(["--limit-files","1"])
        run_step(arguments,output/"logs"/"02_prepare.log")
    if args.stage in ("train","all"):
        run_step(["dominant_hand.train",str(output/"dataset"),"--root",str(ROOT),
                  "--config",str(args.train_config.resolve()),"--jobs",str(args.jobs),
                  "--output",str(output/"training")],output/"logs"/"03_train.log")
    print(f"DONE stage={args.stage}; output={output}",flush=True)
    if args.stage=="smoke":
        print('单ZIP检查已完成。完整实验请将DEFAULT_STAGE改为"all"，或传入--stage all。',flush=True)


if __name__=="__main__": main()
