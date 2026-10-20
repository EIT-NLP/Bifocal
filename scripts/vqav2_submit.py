"""Convert the lmms-eval VQAv2 test-dev output into an EvalAI submission.

lmms-eval leaves test-split answers unnormalized and only covers the 107,394
test-dev questions, while EvalAI expects normalized answers for all 447,793
test2015 questions. This script normalizes every answer and fills the
remaining questions with empty answers.

Usage:
    python scripts/vqav2_submit.py results/budget128
    python scripts/vqav2_submit.py results/budget128 --questions v2_OpenEnded_mscoco_test2015_questions.json
"""
import argparse
import glob
import json
import os

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

TEST2015_SIZE = 447793


def find_prediction(path):
    if os.path.isfile(path):
        return path
    files = sorted(glob.glob(os.path.join(path, "**", "vqav2-test-submission-*.json"), recursive=True),
                   key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"no vqav2-test-submission-*.json under {path}")
    return files[-1]


def test2015_question_ids(questions, dataset):
    if questions:
        with open(questions) as f:
            return [q["question_id"] for q in json.load(f)["questions"]]
    from datasets import load_dataset

    split = load_dataset(dataset, split="test")
    return list(split.select_columns(["question_id"])["question_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="lmms-eval output directory or a vqav2-test-submission-*.json file")
    parser.add_argument("--questions", default=None,
                        help="official v2_OpenEnded_mscoco_test2015_questions.json (avoids downloading test images)")
    parser.add_argument("--dataset", default="lmms-lab/VQAv2")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    source = find_prediction(args.path)
    with open(source) as f:
        raw = json.load(f)
    processor = EvalAIAnswerProcessor()
    predictions = {int(x["question_id"]): processor(str(x["answer"])) for x in raw}

    question_ids = [int(q) for q in test2015_question_ids(args.questions, args.dataset)]
    if len(question_ids) != TEST2015_SIZE:
        raise ValueError(f"expected {TEST2015_SIZE} test2015 questions, got {len(question_ids)}")
    unknown = set(predictions) - set(question_ids)
    if unknown:
        raise ValueError(f"{len(unknown)} predicted question ids are not in test2015")

    output = args.output or os.path.join(os.path.dirname(source), "vqav2_testdev_evalai.json")
    with open(output, "w") as f:
        json.dump([{"question_id": q, "answer": predictions.get(q, "")} for q in question_ids], f)
    print(f"{len(predictions)} predictions from {source}")
    print(f"wrote {len(question_ids)} answers to {output}")


if __name__ == "__main__":
    main()
