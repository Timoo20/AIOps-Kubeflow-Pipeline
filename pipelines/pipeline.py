import kfp
from kfp import dsl

@dsl.component
def preprocess_op():
    import subprocess
    subprocess.run(["python", "src/preprocessing.py"])

@dsl.component
def train_op():
    import subprocess
    subprocess.run(["python", "src/train.py"])

@dsl.component
def evaluate_op():
    import subprocess
    subprocess.run(["python", "src/evaluate.py"])

@dsl.pipeline(name="aiops-kubeflow-pipeline")
def pipeline():
    p = preprocess_op()
    t = train_op()
    e = evaluate_op()

    t.after(p)
    e.after(t)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(pipeline, "pipeline.yaml")
