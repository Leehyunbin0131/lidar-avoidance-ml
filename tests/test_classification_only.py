"""Verification suite for the 3-class (Left / Straight / Right) action pipeline.

Run from the repository root:
    SDL_VIDEODRIVER=dummy uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import os
import sys
import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataset import (  # noqa: E402
    ACTION_LEFT,
    ACTION_NAMES,
    ACTION_RIGHT,
    ACTION_STRAIGHT,
    NUM_ACTIONS,
    DrivingDataset,
    mirror_actions,
)
from expert import baseline_action, generate_expert_dataset, steer_to_action  # noqa: E402
from model import (  # noqa: E402
    BENCHMARK_CSV_HEADER,
    MODEL_TYPES,
    AvoidanceMLP,
    TrainedModel,
    TrainHistory,
    checkpoint_path_for,
    evaluate_models_on_random_maps,
    export_benchmark_csv,
    find_checkpoint,
    legacy_checkpoint_path_for,
    load_checkpoint,
    predict_action,
    prepare_infer_backend,
    save_checkpoint,
    train_model,
)
from simulator import NUM_BEAMS, OBS_DIM, Simulator  # noqa: E402

BENCH_CSV = ROOT / "results" / "benchmark_20maps.csv"

# Shared fixture: one small expert dataset and five quickly-trained models in a temp directory.
_TMP: tempfile.TemporaryDirectory | None = None
TMP_DIR = Path()
DATASET = DrivingDataset()
MODELS: dict = {}


def setUpModule() -> None:
    global _TMP, TMP_DIR
    np.random.seed(0)
    torch.manual_seed(0)
    _TMP = tempfile.TemporaryDirectory(prefix="lidar_cls_test_")
    TMP_DIR = Path(_TMP.name)
    generate_expert_dataset(DATASET, num_samples=800, seed=1)
    DATASET.save(TMP_DIR / "dataset.npz", export_csv=False)
    for m in MODEL_TYPES:
        tm, _ = train_model(DATASET, model_type=m, epochs=2, train_device="cpu",
                            save_path=checkpoint_path_for(TMP_DIR / "models", m))
        MODELS[m] = tm


def tearDownModule() -> None:
    if _TMP is not None:
        _TMP.cleanup()


def _obs(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(OBS_DIM).astype(np.float32)


# ----------------------------------------------------------------------
class TestActionInterface(unittest.TestCase):
    def test_constants(self) -> None:
        self.assertEqual((ACTION_LEFT, ACTION_STRAIGHT, ACTION_RIGHT, NUM_ACTIONS), (0, 1, 2, 3))
        self.assertEqual(tuple(ACTION_NAMES), ("좌회전", "직진", "우회전"))

    def test_steer_to_action(self) -> None:
        self.assertEqual(steer_to_action(-1.0), ACTION_LEFT)
        self.assertEqual(steer_to_action(0.0), ACTION_STRAIGHT)
        self.assertEqual(steer_to_action(1.0), ACTION_RIGHT)

    def test_baseline_action_is_discrete(self) -> None:
        for seed in range(20):
            self.assertIn(baseline_action(_obs(seed)), (0, 1, 2))

    def test_step_action(self) -> None:
        sim = Simulator(noise_std=0.0)
        ag = sim.spawn_robot_state()
        x0, y0 = ag.x, ag.y
        sim.step_action(ag, None, 0.05, add_noise=False)
        self.assertEqual((ag.x, ag.y), (x0, y0))
        sim.step_action(ag, ACTION_STRAIGHT, 0.05, add_noise=False)
        self.assertGreater(ag.distance, 0.0)
        for bad in (3, -1, "x"):
            with self.assertRaises(ValueError):
                sim.step_action(ag, bad, 0.05)

    def test_left_turns_counterclockwise_on_screen(self) -> None:
        sim = Simulator(noise_std=0.0)
        left, right = sim.spawn_robot_state(), sim.spawn_robot_state()
        t0 = left.theta
        sim.step_action(left, ACTION_LEFT, 0.05, add_noise=False)
        sim.step_action(right, ACTION_RIGHT, 0.05, add_noise=False)
        self.assertLess((left.theta - t0 + np.pi) % (2 * np.pi) - np.pi, 0.0)
        self.assertGreater((right.theta - t0 + np.pi) % (2 * np.pi) - np.pi, 0.0)

    def test_manual_step_bump_is_non_terminal(self) -> None:
        sim = Simulator(noise_std=0.0)
        sim.robot.collided = True
        sim.step(ACTION_STRAIGHT, 0.02)
        self.assertGreater(sim.robot.distance, 0.0)


# ----------------------------------------------------------------------
class TestDataset(unittest.TestCase):
    def test_round_trip_obs_and_actions_only(self) -> None:
        ds = DrivingDataset()
        for i in range(9):
            ds.add(_obs(i), i % 3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ds.npz"
            self.assertEqual(ds.save(path, export_csv=True), 9)
            with np.load(path) as data:
                self.assertEqual(set(data.files), {"obs", "actions"})
            with path.with_suffix(".csv").open(encoding="utf-8") as f:
                header = next(csv.reader(f))
            self.assertEqual(header, [f"beam_{i}" for i in range(NUM_BEAMS)]
                             + ["goal_sin", "goal_cos", "goal_dist", "action"])
            loaded = DrivingDataset()
            self.assertEqual(loaded.load(path), 9)
        x0, y0 = ds.to_arrays()
        x1, y1 = loaded.to_arrays()
        np.testing.assert_allclose(x0, x1)
        np.testing.assert_array_equal(y0, y1)
        self.assertEqual(y1.dtype, np.int64)
        self.assertEqual(loaded.class_counts(), (3, 3, 3))

    def test_legacy_npz_with_vels_loads(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "legacy.npz"
            np.savez_compressed(path, obs=np.random.rand(5, OBS_DIM).astype(np.float32),
                                vels=np.random.rand(5, 2).astype(np.float32),
                                actions=np.array([0, 1, 2, 1, 0], dtype=np.int64))
            ds = DrivingDataset()
            self.assertEqual(ds.load(path), 5)
            self.assertEqual(ds.class_counts(), (2, 2, 1))

    def test_missing_actions_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "noact.npz"
            np.savez_compressed(path, obs=np.random.rand(4, OBS_DIM).astype(np.float32))
            self.assertEqual(DrivingDataset().load(path), 0)

    def test_mirror_actions(self) -> None:
        y = np.array([0, 1, 2, 2, 0], dtype=np.int64)
        out = mirror_actions(y)
        np.testing.assert_array_equal(out, [2, 1, 0, 0, 2])
        self.assertEqual(out.dtype, y.dtype)
        self.assertEqual(out.shape, y.shape)
        np.testing.assert_array_equal(y, [0, 1, 2, 2, 0])  # input untouched

    def test_invalid_samples_raise(self) -> None:
        ds = DrivingDataset()
        for bad in (3, -1):
            with self.assertRaises(ValueError):
                ds.add(_obs(), bad)
        with self.assertRaises(ValueError):
            ds.add(np.zeros(OBS_DIM - 1, dtype=np.float32), ACTION_LEFT)
        self.assertEqual(len(ds), 0)


# ----------------------------------------------------------------------
class TestCheckpoints(unittest.TestCase):
    def _torch_payload(self, objective: str | None) -> dict:
        payload = {"model_type": "mlp", "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS,
                   "state_dict": AvoidanceMLP().state_dict(), "val_metric": 50.0}
        if objective is not None:
            payload["task_type"] = objective
        return payload

    def test_legacy_regression_payload_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            pt = Path(d) / "goal_mlp_reg.pt"
            torch.save(self._torch_payload("regression"), pt)
            with self.assertRaises(ValueError):
                load_checkpoint(pt)
            jb = Path(d) / "goal_linear_reg.joblib"
            joblib.dump({**MODELS["linear"].__dict__, "estimator": MODELS["linear"].estimator,
                         "task_type": "regression"}, jb)
            with self.assertRaises(ValueError):
                load_checkpoint(jb)

    def test_non_classifier_estimator_rejected(self) -> None:
        from sklearn.linear_model import LinearRegression

        with tempfile.TemporaryDirectory() as d:
            jb = Path(d) / "goal_linear.joblib"
            est = LinearRegression().fit(np.random.rand(10, OBS_DIM), np.random.rand(10))
            joblib.dump({"model_type": "linear", "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS, "estimator": est}, jb)
            with self.assertRaises(ValueError):
                load_checkpoint(jb)

    def test_legacy_classification_payload_loads(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            models_dir = Path(d)
            legacy = legacy_checkpoint_path_for(models_dir, "mlp")
            self.assertEqual(legacy.name, "goal_mlp_cls.pt")
            torch.save(self._torch_payload("classification"), legacy)
            self.assertEqual(find_checkpoint(models_dir, "mlp"), legacy)
            tm = load_checkpoint(legacy)
            self.assertEqual(tm.model_type, "mlp")
            # Migration writes the canonical name, which then takes precedence.
            save_checkpoint(tm, checkpoint_path_for(models_dir, "mlp"))
            self.assertEqual(find_checkpoint(models_dir, "mlp"), models_dir / "goal_mlp.pt")
            ckpt = torch.load(models_dir / "goal_mlp.pt", weights_only=True)
            self.assertNotIn("task_type", ckpt)
            self.assertIsNone(find_checkpoint(models_dir, "rf"))

    def test_checkpoint_names(self) -> None:
        d = Path("m")
        self.assertEqual(checkpoint_path_for(d, "cnn1d").name, "goal_cnn1d.pt")
        self.assertEqual(checkpoint_path_for(d, "rf").name, "goal_rf.joblib")
        self.assertEqual(legacy_checkpoint_path_for(d, "knn").name, "goal_knn_cls.joblib")


# ----------------------------------------------------------------------
class TestModels(unittest.TestCase):
    def test_all_models_round_trip_and_predict(self) -> None:
        obs_batch, _ = DATASET.to_arrays()
        for m in MODEL_TYPES:
            with self.subTest(model=m):
                path = find_checkpoint(TMP_DIR / "models", m)
                self.assertEqual(path, checkpoint_path_for(TMP_DIR / "models", m))
                loaded = load_checkpoint(path)
                self.assertEqual(loaded.model_type, m)
                self.assertAlmostEqual(loaded.val_metric, MODELS[m].val_metric, places=4)
                for obs in obs_batch[:25]:
                    action, probs = predict_action(loaded, obs)
                    self.assertIsInstance(action, int)
                    self.assertIn(action, (0, 1, 2))
                    self.assertEqual(probs.shape, (3,))
                    self.assertEqual(probs.dtype, np.float32)
                    self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
                    self.assertEqual(action, int(np.argmax(probs)))
                    a_orig, p_orig = predict_action(MODELS[m], obs)
                    self.assertEqual(a_orig, action)
                    np.testing.assert_allclose(p_orig, probs, atol=1e-5)

    def test_torch_nets_output_logits(self) -> None:
        x = torch.zeros(4, OBS_DIM)
        for m in ("mlp", "cnn1d"):
            with torch.no_grad():
                out = MODELS[m].estimator.to("cpu")(x)
            self.assertEqual(tuple(out.shape), (4, NUM_ACTIONS))

    def test_openvino_cpu_parity(self) -> None:
        from accelerator import detect_hardware

        if "CPU" not in detect_hardware().ov_devices:
            self.skipTest("OpenVINO CPU device not available")
        x, _ = DATASET.to_arrays()
        x = x[:200]
        for m in ("mlp", "cnn1d"):
            with self.subTest(model=m):
                tm = MODELS[m]
                prepare_infer_backend(tm, "torch_cpu")
                ref = [predict_action(tm, o)[0] for o in x]
                export_dir = TMP_DIR / "ov"
                prepare_infer_backend(tm, "ov_cpu", export_dir=export_dir)
                try:
                    self.assertTrue((export_dir / f"ov_{m}.xml").exists())
                    out = [predict_action(tm, o) for o in x]
                finally:
                    prepare_infer_backend(tm, "torch_cpu")
                match = np.mean([a == r for (a, _), r in zip(out, ref)])
                self.assertGreaterEqual(match, 0.99)
                for _, probs in out:
                    self.assertEqual(probs.shape, (3,))
                    self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_benchmark_and_csv(self) -> None:
        entries = evaluate_models_on_random_maps(MODELS, num_maps=2, seed_start=1000)
        self.assertEqual([e.key for e in entries], ["baseline", *MODEL_TYPES])
        for e in entries:
            self.assertEqual(e.goals + e.crashes + e.timeouts, 2)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bench.csv"
            export_benchmark_csv(entries, path)
            with path.open(encoding="utf-8") as f:
                rows = list(csv.reader(f))
        self.assertEqual(tuple(rows[0]), BENCHMARK_CSV_HEADER)
        self.assertEqual(len(rows), 1 + len(entries))


# ----------------------------------------------------------------------
class TestExpertGeneration(unittest.TestCase):
    def test_generate_expert_dataset(self) -> None:
        np.random.seed(0)
        ds = DrivingDataset()
        res = generate_expert_dataset(ds, 300, seed=0)
        self.assertEqual(res["added"], 300)
        self.assertEqual(len(ds), 300)
        counts = ds.class_counts()
        self.assertEqual(sum(counts), 300)
        self.assertTrue(all(c > 0 for c in counts), counts)


# ----------------------------------------------------------------------
class TestInterpretation(unittest.TestCase):
    def test_reads_new_benchmark_csv(self) -> None:
        from interpretation import read_saved_report, result_markdown

        report = read_saved_report(BENCH_CSV)
        self.assertIsNotNone(report)
        self.assertEqual(len(report.entries), 6)
        self.assertEqual(report.entries[0].key, "baseline")
        md = result_markdown(report)
        self.assertIn("검증 정확도", md)
        self.assertNotIn("MAE", md)
        self.assertNotIn("회귀", md)
        self.assertNotIn("지만,", md)
        self.assertTrue(md.startswith(f"### 저장된 평가 · 같은 맵 {report.entries[0].num_maps}개"))

    def test_rejects_legacy_mae_csv(self) -> None:
        from interpretation import read_saved_report

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "legacy.csv"
            path.write_text(
                "model,success_rate_pct,goals,crashes,timeouts,avg_goal_time_s,val_mae,infer_ms\n"
                "라이다 회피 기준선,85.0,17,0,3,8.13,-,0.010\n"
                "MLP (DL),100.0,20,0,0,4.82,0.0561,0.250\n",
                encoding="utf-8",
            )
            self.assertIsNone(read_saved_report(path))

    def test_outcome_sentences_have_no_dangling_commas(self) -> None:
        from interpretation import _outcome_sentence
        from model import BenchmarkEntry

        for goals, crashes, timeouts in ((0, 0, 20), (10, 0, 10), (10, 10, 0), (5, 5, 10), (0, 5, 15), (20, 0, 0)):
            e = BenchmarkEntry("mlp", "MLP (DL)", 20, goals, crashes, timeouts, goals * 5.0, 5.0, 80.0, 0.1)
            sentence = _outcome_sentence("MLP", e)
            self.assertNotIn("지만,", sentence)
            self.assertNotIn("고,", sentence)


# ----------------------------------------------------------------------
# Regression tests for the adversarial review findings (#1-#10).
# ----------------------------------------------------------------------
def _ov_cpu_available() -> bool:
    from accelerator import detect_hardware

    return "CPU" in detect_hardware().ov_devices


def _bad_observations() -> dict:
    nan_obs = _obs(1)
    nan_obs[3] = np.nan
    inf_obs = _obs(2)
    inf_obs[OBS_DIM - 1] = np.inf
    return {
        "nan": nan_obs,
        "inf": inf_obs,
        "short": np.zeros(OBS_DIM - 1, dtype=np.float32),
        "long": np.zeros(OBS_DIM + 1, dtype=np.float32),
        "row": np.zeros((1, OBS_DIM), dtype=np.float32),
        "batch": np.zeros((2, OBS_DIM), dtype=np.float32),
        "column": np.zeros((OBS_DIM, 1), dtype=np.float32),
    }


BAD_ACTIONS = (True, False, np.bool_(True), 1.0, 1.8, -0.9, 0.0, np.float32(1.0), np.float64(2.0), "1", 3, -1,
               np.int64(3), [1])


class TestReviewF1OpenVINOConcurrency(unittest.TestCase):
    def test_runner_is_safe_under_concurrent_callers(self) -> None:
        from accelerator import compile_torch_to_openvino

        if not _ov_cpu_available():
            self.skipTest("OpenVINO CPU device not available")
        net = MODELS["mlp"].estimator.to("cpu").eval()
        runner = compile_torch_to_openvino(net, obs_dim=OBS_DIM, ov_device="CPU")
        x = np.stack([_obs(i) for i in range(6)]).astype(np.float32)
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy()
        errors: list = []
        mismatches: list = []

        def worker(i: int) -> None:
            for _ in range(250):
                try:
                    out = runner.predict(x[i : i + 1])
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(repr(exc))
                    return
                if not np.allclose(out[0], ref[i], atol=1e-4):
                    mismatches.append(i)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(mismatches, [])


class TestReviewF3ObservationValidation(unittest.TestCase):
    def test_dataset_add_rejects_invalid_observations(self) -> None:
        ds = DrivingDataset()
        for name, bad in _bad_observations().items():
            with self.subTest(obs=name), self.assertRaises(ValueError):
                ds.add(bad, ACTION_STRAIGHT)
        self.assertEqual(len(ds), 0)

    def test_dataset_load_rejects_non_finite_observations(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nan.npz"
            obs = np.random.rand(6, OBS_DIM).astype(np.float32)
            obs[2, 5] = np.nan
            np.savez_compressed(path, obs=obs, actions=np.array([0, 1, 2, 0, 1, 2], dtype=np.int64))
            ds = DrivingDataset()
            ds.add(_obs(), ACTION_LEFT)
            self.assertEqual(ds.load(path), 0)
            self.assertEqual(len(ds), 1)  # a rejected file leaves the current buffer intact

    def test_train_model_rejects_non_finite_observations(self) -> None:
        for m in ("mlp", "rf"):
            with self.subTest(model=m):
                ds = DrivingDataset()
                for i in range(30):
                    ds.add(_obs(i), i % 3)
                ds._obs[4][7] = np.nan  # corrupt a stored row behind the validated API
                with self.assertRaises(ValueError):
                    train_model(ds, model_type=m, epochs=1, train_device="cpu")

    def test_predict_action_rejects_invalid_observations(self) -> None:
        for m in MODEL_TYPES:
            for name, bad in _bad_observations().items():
                with self.subTest(model=m, obs=name), self.assertRaises(ValueError):
                    predict_action(MODELS[m], bad)

    def test_predict_action_rejects_non_finite_model_output(self) -> None:
        net = AvoidanceMLP()
        with torch.no_grad():
            for prm in net.parameters():
                prm.fill_(float("nan"))
        tm = TrainedModel(model_type="mlp", obs_dim=OBS_DIM, num_beams=NUM_BEAMS, backend="torch", estimator=net.eval())
        with self.assertRaises(ValueError):
            predict_action(tm, _obs())

    def test_baseline_action_rejects_invalid_observations(self) -> None:
        for name, bad in _bad_observations().items():
            with self.subTest(obs=name), self.assertRaises(ValueError):
                baseline_action(bad)
        with self.assertRaises(ValueError):
            baseline_action(np.zeros(10 + 3, dtype=np.float32), num_beams=11)
        self.assertIn(baseline_action(np.ones(11 + 3, dtype=np.float32), num_beams=11), (0, 1, 2))


class TestReviewF4ActionValidation(unittest.TestCase):
    def test_validate_action(self) -> None:
        from simulator import validate_action

        for good, expected in ((0, 0), (1, 1), (2, 2), (np.int64(2), 2), (np.int32(0), 0), (np.uint8(1), 1)):
            with self.subTest(action=good):
                out = validate_action(good)
                self.assertEqual(out, expected)
                self.assertIs(type(out), int)
        for bad in BAD_ACTIONS:
            with self.subTest(action=bad), self.assertRaises(ValueError):
                validate_action(bad)

    def test_step_action_rejects_bool_and_fractional_actions(self) -> None:
        sim = Simulator(noise_std=0.0)
        for bad in BAD_ACTIONS:
            ag = sim.spawn_robot_state()
            with self.subTest(action=bad):
                with self.assertRaises(ValueError):
                    sim.step_action(ag, bad, 0.05, add_noise=False)
                self.assertEqual(ag.distance, 0.0)
                self.assertEqual(ag.elapsed, 0.0)
        ag = sim.spawn_robot_state()
        sim.step_action(ag, np.int64(ACTION_STRAIGHT), 0.05, add_noise=False)
        self.assertGreater(ag.distance, 0.0)

    def test_dataset_add_rejects_bool_and_fractional_actions(self) -> None:
        ds = DrivingDataset()
        for bad in BAD_ACTIONS:
            with self.subTest(action=bad), self.assertRaises(ValueError):
                ds.add(_obs(), bad)
        self.assertEqual(len(ds), 0)
        ds.add(_obs(), np.int64(ACTION_RIGHT))
        self.assertEqual(ds.class_counts(), (0, 0, 1))

    def test_dataset_load_rejects_non_integer_action_arrays(self) -> None:
        cases = {
            "fractional": np.array([-0.5, 1.9, 2.2], dtype=np.float64),
            "integral_floats": np.array([0.0, 1.0, 2.0], dtype=np.float32),
            "bool": np.array([True, False, True]),
            "strings": np.array(["0", "1", "2"]),
        }
        with tempfile.TemporaryDirectory() as d:
            for name, actions in cases.items():
                with self.subTest(actions=name):
                    path = Path(d) / f"{name}.npz"
                    np.savez_compressed(path, obs=np.random.rand(3, OBS_DIM).astype(np.float32), actions=actions)
                    self.assertEqual(DrivingDataset().load(path), 0)
            ok = Path(d) / "ok.npz"
            np.savez_compressed(ok, obs=np.random.rand(3, OBS_DIM).astype(np.float32),
                                actions=np.array([0, 1, 2], dtype=np.int32))
            self.assertEqual(DrivingDataset().load(ok), 3)


class TestReviewF5BalancedValidationLoss(unittest.TestCase):
    @staticmethod
    def _straight_only_validation_dataset() -> DrivingDataset:
        val = set(np.random.default_rng(42).permutation(20)[:4].tolist())
        ds = DrivingDataset()
        for i in range(20):
            ds.add(_obs(100 + i), ACTION_STRAIGHT if i in val else (ACTION_LEFT if i % 2 else ACTION_RIGHT))
        return ds

    def test_val_loss_finite_when_val_class_missing_from_train(self) -> None:
        ds = self._straight_only_validation_dataset()
        for m in ("mlp", "cnn1d"):
            with self.subTest(model=m):
                _, hist = train_model(ds, model_type=m, epochs=4, class_balance=True, train_device="cpu")
                self.assertEqual(len(hist.val_losses), 4)
                self.assertTrue(np.all(np.isfinite(hist.val_losses)), hist.val_losses)
                self.assertTrue(np.all(np.isfinite(hist.train_losses)), hist.train_losses)
                best = int(np.argmin(hist.val_losses))
                self.assertEqual(hist.best_val_metric, hist.val_metrics[best])

    def test_balanced_class_weights_always_finite_positive(self) -> None:
        from model import _balanced_class_weights

        for y in (np.array([0, 0, 2, 2]), np.array([1, 1, 1, 0]), np.array([0, 1, 2, 2, 2])):
            with self.subTest(y=y.tolist()):
                w = _balanced_class_weights(y)
                self.assertEqual(w.shape, (NUM_ACTIONS,))
                self.assertTrue(np.all(np.isfinite(w)))
                self.assertTrue(np.all(w > 0))


class TestReviewF6CheckpointValidation(unittest.TestCase):
    @staticmethod
    def _dump(path: Path, model_type: str, est) -> Path:
        joblib.dump({"model_type": model_type, "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS, "estimator": est}, path)
        return path

    def test_estimator_class_must_match_model_type(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            rf = MODELS["rf"].estimator
            linear = MODELS["linear"].estimator
            for name, (model_type, est) in {
                "rf_as_linear": ("linear", rf),
                "linear_as_knn": ("knn", linear),
                "knn_as_rf": ("rf", MODELS["knn"].estimator),
                "unknown_type": ("svm", linear),
                "torch_type_in_joblib": ("mlp", linear),
            }.items():
                with self.subTest(case=name), self.assertRaises(ValueError):
                    load_checkpoint(self._dump(Path(d) / f"{name}.joblib", model_type, est))

    def test_feature_count_and_classes_validated(self) -> None:
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            narrow = LogisticRegression(max_iter=200).fit(rng.random((30, 10)), np.arange(30) % 3)
            with self.assertRaises(ValueError):
                load_checkpoint(self._dump(Path(d) / "narrow.joblib", "linear", narrow))
            alien = LogisticRegression(max_iter=200).fit(rng.random((30, OBS_DIM)), 3 + np.arange(30) % 2)
            with self.assertRaises(ValueError):
                load_checkpoint(self._dump(Path(d) / "alien.joblib", "linear", alien))
            binary = LogisticRegression(max_iter=200).fit(rng.random((30, OBS_DIM)), 2 * (np.arange(30) % 2))
            tm = load_checkpoint(self._dump(Path(d) / "binary.joblib", "linear", binary))
            self.assertEqual(tm.model_type, "linear")


class TestReviewF7LatencyRefresh(unittest.TestCase):
    def test_switch_back_to_torch_remeasures_latency(self) -> None:
        tm = load_checkpoint(checkpoint_path_for(TMP_DIR / "models", "mlp"))
        if _ov_cpu_available():
            prepare_infer_backend(tm, "ov_cpu")
            self.assertEqual(tm.active_infer_backend, "ov_cpu")
        tm.infer_ms = 999.0  # stale accelerator timing
        prepare_infer_backend(tm, "torch_cpu")
        self.assertEqual(tm.active_infer_backend, "torch_cpu")
        self.assertNotEqual(tm.infer_ms, 999.0)
        self.assertGreater(tm.infer_ms, 0.0)
        self.assertLess(tm.infer_ms, 100.0)


class TestReviewF8EmptyDatasetSave(unittest.TestCase):
    def test_empty_save_overwrites_stale_csv(self) -> None:
        ds = DrivingDataset()
        for i in range(4):
            ds.add(_obs(i), i % 3)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ds.npz"
            ds.save(path, export_csv=True)
            ds.clear()
            self.assertEqual(ds.save(path, export_csv=True), 0)
            with path.with_suffix(".csv").open(encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows, [[f"beam_{i}" for i in range(NUM_BEAMS)]
                                    + ["goal_sin", "goal_cos", "goal_dist", "action"]])
            self.assertEqual(DrivingDataset().load(path), 0)


class TestReviewF9FigureLinearCoefficients(unittest.TestCase):
    @staticmethod
    def _script():
        path = ROOT / "scripts" / "generate_guide_figures.py"
        name = "generate_guide_figures_under_test"
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses in the script resolve their module through sys.modules
        spec.loader.exec_module(module)
        return module

    def test_binary_and_three_class_linear_coefficients(self) -> None:
        from sklearn.linear_model import LogisticRegression

        script = self._script()
        rng = np.random.default_rng(3)
        x = rng.random((40, OBS_DIM))
        binary = LogisticRegression(max_iter=300).fit(x, np.where(np.arange(40) % 2, ACTION_RIGHT, ACTION_LEFT))
        self.assertEqual(binary.coef_.shape, (1, OBS_DIM))
        w = script.linear_class_coefficients(binary)
        self.assertEqual(w.shape, (NUM_ACTIONS, OBS_DIM))
        np.testing.assert_allclose(w[ACTION_RIGHT], binary.coef_[0])
        np.testing.assert_allclose(w[ACTION_LEFT], -binary.coef_[0])
        np.testing.assert_array_equal(w[ACTION_STRAIGHT], 0.0)

        three = LogisticRegression(max_iter=300).fit(x, np.arange(40) % 3)
        np.testing.assert_allclose(script.linear_class_coefficients(three), three.coef_)

        mixed = LogisticRegression(max_iter=300).fit(x, np.where(np.arange(40) % 2, ACTION_STRAIGHT, ACTION_RIGHT))
        w = script.linear_class_coefficients(mixed)
        np.testing.assert_allclose(w[ACTION_STRAIGHT], -mixed.coef_[0])
        np.testing.assert_allclose(w[ACTION_RIGHT], mixed.coef_[0])
        np.testing.assert_array_equal(w[ACTION_LEFT], 0.0)


def _figure_texts(fig) -> list:
    texts = [fig._suptitle.get_text()] if fig._suptitle is not None else []
    texts += [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        texts.append(ax.get_title())
        texts += [t.get_text() for t in ax.texts]
    return texts


class TestReviewF10VisualizationLabels(unittest.TestCase):
    def test_scatter2d_labels_surrogate_classifier(self) -> None:
        import matplotlib.pyplot as plt

        from visualizer import build_figure

        for model, trained, surrogate in (("cnn1d", {}, "MLPClassifier"), ("mlp", {}, "MLPClassifier"),
                                          ("linear", MODELS, "LogisticRegression")):
            with self.subTest(model=model):
                fig = build_figure("scatter2d", DATASET, trained, model, [])
                try:
                    boundary_title = fig.axes[1].get_title()
                    joined = "\n".join(_figure_texts(fig))
                finally:
                    plt.close(fig)
                self.assertIn("대리", boundary_title)
                self.assertIn(surrogate, boundary_title)
                self.assertNotIn("1D CNN", boundary_title)
                self.assertNotIn("1D CNN", fig._suptitle.get_text())
                if model in trained:
                    self.assertNotIn("체크포인트 없음", joined)
                else:
                    self.assertIn("체크포인트 없음", joined)

    def test_classification_chart_uses_full_validation_split(self) -> None:
        import matplotlib.pyplot as plt

        from visualizer import build_figure

        rng = np.random.default_rng(11)
        ds = DrivingDataset()
        obs = rng.random((10_000, OBS_DIM)).astype(np.float32)
        labels = np.where(obs[:, 0] > 0.5, ACTION_LEFT, ACTION_RIGHT)
        noisy = rng.random(len(labels)) < 0.3
        labels[noisy] = rng.integers(0, NUM_ACTIONS, int(noisy.sum()))
        for o, a in zip(obs, labels):
            ds.add(o, int(a))
        tm, _ = train_model(ds, model_type="linear", mirror_augment=False, train_device="cpu")
        fig = build_figure("classification", ds, {"linear": tm}, "linear", [])
        try:
            title = fig._suptitle.get_text()
            counts = [int(t.get_text().split("\n")[0]) for t in fig.axes[0].texts]
        finally:
            plt.close(fig)
        self.assertEqual(len(counts), 9)
        self.assertEqual(sum(counts), 2_000)  # full 20% validation split, no 1,500-sample cap
        self.assertIn(f"전체 정확도: {tm.val_metric:.1f}%", title)


# ----------------------------------------------------------------------
# Regression tests for the sign-off review findings (S1-S8).
# ----------------------------------------------------------------------
class TestSignoffS2ObservationDtypes(unittest.TestCase):
    def test_validate_observation_rejects_non_real_dtypes(self) -> None:
        from simulator import validate_observation

        cases = {
            "complex": np.ones(OBS_DIM, dtype=np.complex64),
            "complex_zero_imag": np.zeros(OBS_DIM, dtype=np.complex128),
            "bool": np.ones(OBS_DIM, dtype=bool),
            "strings": np.array(["1"] * OBS_DIM),
            "object": np.array([0.5] * OBS_DIM, dtype=object),
            "list_of_str": ["0.5"] * OBS_DIM,
        }
        for name, obs in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                validate_observation(obs)
        for ok in (np.zeros(OBS_DIM, dtype=np.float64), np.arange(OBS_DIM, dtype=np.int32), [0.25] * OBS_DIM):
            out = validate_observation(ok)
            self.assertEqual(out.dtype, np.float32)
            self.assertEqual(out.shape, (OBS_DIM,))

    def test_dataset_and_predict_reject_complex_observations(self) -> None:
        obs = _obs(4).astype(np.complex64)
        with self.assertRaises(ValueError):
            DrivingDataset().add(obs, ACTION_LEFT)
        with self.assertRaises(ValueError):
            predict_action(MODELS["linear"], obs)
        with self.assertRaises(ValueError):
            baseline_action(obs)


class TestSignoffS3CheckpointIdentity(unittest.TestCase):
    def test_expected_model_type_is_enforced(self) -> None:
        linear_path = checkpoint_path_for(TMP_DIR / "models", "linear")
        with self.assertRaises(ValueError):
            load_checkpoint(linear_path, expected_model_type="knn")
        self.assertEqual(load_checkpoint(linear_path, expected_model_type="linear").model_type, "linear")
        mlp_path = checkpoint_path_for(TMP_DIR / "models", "mlp")
        with self.assertRaises(ValueError):
            load_checkpoint(mlp_path, expected_model_type="cnn1d")

    def test_nan_weights_fail_at_load(self) -> None:
        import copy

        with tempfile.TemporaryDirectory() as d:
            state = {k: v.detach().cpu().clone() for k, v in MODELS["mlp"].estimator.state_dict().items()}
            first = next(k for k, v in state.items() if v.is_floating_point())
            state[first].fill_(float("nan"))
            pt = Path(d) / "goal_mlp.pt"
            torch.save({"model_type": "mlp", "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS, "state_dict": state}, pt)
            with self.assertRaises(ValueError):
                load_checkpoint(pt)
            est = copy.deepcopy(MODELS["linear"].estimator)
            est.named_steps["clf"].coef_[:] = np.nan
            jl = Path(d) / "goal_linear.joblib"
            joblib.dump({"model_type": "linear", "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS, "estimator": est}, jl)
            with self.assertRaises(ValueError):
                load_checkpoint(jl)

    def test_pipeline_inner_feature_mismatch_fails_at_load(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(5)
        scaler = StandardScaler().fit(rng.random((30, OBS_DIM)))
        clf = LogisticRegression(max_iter=200).fit(rng.random((30, 2)), np.arange(30) % 3)
        est = Pipeline([("scaler", scaler), ("clf", clf)])
        self.assertEqual(est.n_features_in_, OBS_DIM)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "goal_linear.joblib"
            joblib.dump({"model_type": "linear", "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS, "estimator": est}, path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_fallback_helper_and_figure_script_use_legacy_sibling(self) -> None:
        import shutil

        from model import load_checkpoint_with_fallback

        script = TestReviewF9FigureLinearCoefficients._script()
        with tempfile.TemporaryDirectory() as d:
            models_dir = Path(d)
            # Canonical rf holds a valid *linear* checkpoint; canonical mlp has NaN weights.
            shutil.copy(checkpoint_path_for(TMP_DIR / "models", "linear"), checkpoint_path_for(models_dir, "rf"))
            shutil.copy(checkpoint_path_for(TMP_DIR / "models", "rf"), legacy_checkpoint_path_for(models_dir, "rf"))
            state = {k: v.detach().cpu().clone() for k, v in MODELS["mlp"].estimator.state_dict().items()}
            for v in state.values():
                if v.is_floating_point():
                    v.fill_(float("nan"))
            torch.save({"model_type": "mlp", "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS, "state_dict": state},
                       checkpoint_path_for(models_dir, "mlp"))
            shutil.copy(checkpoint_path_for(TMP_DIR / "models", "mlp"), legacy_checkpoint_path_for(models_dir, "mlp"))
            for key in ("rf", "mlp"):
                with self.subTest(key=key):
                    tm, errors = load_checkpoint_with_fallback(models_dir, key)
                    self.assertIsNotNone(tm)
                    self.assertEqual(tm.model_type, key)
                    self.assertEqual(len(errors), 1)
                    self.assertIn(checkpoint_path_for(models_dir, key).name, errors[0])
                    with mock.patch.object(script, "MODELS_DIR", models_dir):
                        self.assertEqual(script.load_model(key).model_type, key)
            tm, errors = load_checkpoint_with_fallback(models_dir, "knn")
            self.assertIsNone(tm)
            self.assertEqual(errors, [])
            with mock.patch.object(script, "MODELS_DIR", models_dir), self.assertRaises(FileNotFoundError):
                script.load_model("knn")


class TestSignoffS5CorruptDatasetFiles(unittest.TestCase):
    def test_corrupt_or_object_npz_returns_zero_and_keeps_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / "good.npz"
            src = DrivingDataset()
            for i in range(5):
                src.add(_obs(i), i % 3)
            src.save(good, export_csv=False)
            blob = good.read_bytes()
            files = {}
            files["object_obs"] = Path(d) / "object_obs.npz"
            np.savez(files["object_obs"], obs=np.array([[0.1] * OBS_DIM] * 3, dtype=object), actions=np.array([0, 1, 2]))
            files["object_actions"] = Path(d) / "object_actions.npz"
            np.savez(files["object_actions"], obs=np.zeros((3, OBS_DIM), np.float32),
                     actions=np.array([0, 1, 2], dtype=object))
            files["garbage"] = Path(d) / "garbage.npz"
            files["garbage"].write_bytes(b"this is not a numpy archive at all")
            files["empty"] = Path(d) / "empty.npz"
            files["empty"].write_bytes(b"")
            files["truncated"] = Path(d) / "truncated.npz"
            files["truncated"].write_bytes(blob[: len(blob) // 2])
            for name, path in files.items():
                with self.subTest(case=name):
                    ds = DrivingDataset()
                    for i in range(3):
                        ds.add(_obs(10 + i), ACTION_STRAIGHT)
                    before = ds.to_arrays()
                    self.assertEqual(ds.load(path), 0)
                    after = ds.to_arrays()
                    self.assertEqual(len(ds), 3)
                    np.testing.assert_array_equal(before[0], after[0])
                    np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(DrivingDataset().load(good), 5)


class TestSignoffS6OpenVINOExportRestoresDevice(unittest.TestCase):
    def test_export_failure_restores_original_device(self) -> None:
        import types

        import accelerator

        if not accelerator.HAS_OPENVINO:
            self.skipTest("OpenVINO not installed")

        class FakeNet:
            def __init__(self) -> None:
                self.device = torch.device("cuda")
                self.moves: list = []

            def parameters(self):
                yield types.SimpleNamespace(device=self.device)

            def to(self, dev):
                self.device = torch.device(dev)
                self.moves.append(self.device.type)
                return self

            def eval(self):
                return self

        net = FakeNet()
        with mock.patch.object(accelerator.ov, "convert_model", side_effect=RuntimeError("simulated export failure")):
            with self.assertRaises(RuntimeError):
                accelerator.compile_torch_to_openvino(net, obs_dim=OBS_DIM, ov_device="CPU")
        self.assertEqual(net.device.type, "cuda")
        self.assertEqual(net.moves[0], "cpu")


class TestSignoffS7BareLogisticRegression(unittest.TestCase):
    def test_visualizer_and_script_accept_bare_or_pipeline_linear(self) -> None:
        import dataclasses

        import matplotlib.pyplot as plt
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from visualizer import build_figure

        script = TestReviewF9FigureLinearCoefficients._script()
        x, _ = DATASET.to_arrays()
        rng = np.random.default_rng(9)
        labels = {"binary": np.where(np.arange(len(x)) % 2, ACTION_RIGHT, ACTION_LEFT),
                  "three": rng.integers(0, NUM_ACTIONS, len(x))}
        for name, y in labels.items():
            bare = LogisticRegression(max_iter=300).fit(x, y)
            piped = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=300))]).fit(x, y)
            for kind, est in (("bare", bare), ("pipeline", piped)):
                with self.subTest(labels=name, kind=kind):
                    tm = dataclasses.replace(MODELS["linear"], estimator=est)
                    fig = build_figure("importance", DATASET, {"linear": tm}, "linear", [])
                    try:
                        self.assertIn("로지스틱 회귀", fig._suptitle.get_text())
                    finally:
                        plt.close(fig)
                    w = script.linear_class_coefficients(est)
                    self.assertEqual(w.shape, (NUM_ACTIONS, OBS_DIM))
                    inner = est.named_steps["clf"] if hasattr(est, "named_steps") else est
                    if name == "three":
                        np.testing.assert_allclose(w, inner.coef_)
                    else:
                        np.testing.assert_allclose(w[ACTION_RIGHT], inner.coef_[0])


class TestSignoffS8VisualizationTransparency(unittest.TestCase):
    def test_confusion_chart_labels_dataset_drift(self) -> None:
        import matplotlib.pyplot as plt

        from visualizer import build_figure

        rng = np.random.default_rng(21)
        small = DrivingDataset()
        for o, a in zip(*DATASET.to_arrays()):
            if rng.random() < 0.5:
                small.add(o, int(a))
        tm, _ = train_model(small, model_type="linear", mirror_augment=False, train_device="cpu")
        drift_note = "현재 데이터셋 분할 평가 (학습 데이터와 다름 · 체크포인트 기록"
        for ds, drifted in ((small, False), (DATASET, True)):
            with self.subTest(drifted=drifted):
                fig = build_figure("classification", ds, {"linear": tm}, "linear", [])
                try:
                    joined = "\n".join(_figure_texts(fig))
                finally:
                    plt.close(fig)
                if drifted:
                    self.assertIn(drift_note, joined)
                    self.assertIn(f"{tm.val_metric:.1f}%", joined)
                else:
                    self.assertNotIn(drift_note, joined)
                    self.assertIn(f"전체 정확도: {tm.val_metric:.1f}%", joined)
        # A checkpoint saved/loaded keeps its training-set size, so the chart stays honest after restart.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "goal_linear.joblib"
            save_checkpoint(tm, path)
            loaded = load_checkpoint(path, expected_model_type="linear")
        self.assertEqual(loaded.num_samples, len(small))

    def test_scatter2d_single_class_does_not_claim_a_boundary(self) -> None:
        import matplotlib.pyplot as plt

        from visualizer import build_figure

        ds = DrivingDataset()
        for i in range(60):
            ds.add(_obs(100 + i), ACTION_STRAIGHT)
        fig = build_figure("scatter2d", ds, MODELS, "linear", [])
        try:
            boundary_title = fig.axes[1].get_title()
            joined = "\n".join(_figure_texts(fig))
        finally:
            plt.close(fig)
        self.assertIn("결정 경계 미표시 (2개 이상 클래스 필요)", joined)
        self.assertNotIn("새로 맞춘", joined)
        self.assertNotIn("결정 경계 ·", boundary_title)


class TestFinalSignoffRemediation(unittest.TestCase):
    """Fix cycle 3: corrupt deflate NPZ, staged OpenVINO runner, non-finite parameters, provenance, pipelines."""

    def test_corrupt_compressed_npz_returns_zero_and_keeps_buffer(self) -> None:
        import zipfile

        with tempfile.TemporaryDirectory() as d:
            good = Path(d) / "good.npz"
            rng = np.random.default_rng(3)
            np.savez_compressed(good, obs=rng.random((60, OBS_DIM)).astype(np.float32),
                                actions=np.arange(60, dtype=np.int64) % NUM_ACTIONS)
            blob = bytearray(good.read_bytes())
            with zipfile.ZipFile(good) as zf:
                info = zf.getinfo("obs.npy")
            start = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
            blob[start + 10:start + 60] = b"\xff" * 50  # break the deflate stream, not the zip directory
            bad = Path(d) / "bad_deflate.npz"
            bad.write_bytes(bytes(blob))
            ds = DrivingDataset()
            for i in range(3):
                ds.add(_obs(20 + i), ACTION_LEFT)
            before = ds.to_arrays()
            self.assertEqual(ds.load(bad), 0)
            after = ds.to_arrays()
            np.testing.assert_array_equal(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(DrivingDataset().load(good), 60)

    def test_openvino_runner_is_verified_before_activation(self) -> None:
        import dataclasses
        import types

        import model as model_module

        class _Runner:
            compile_ms = 4.0

            def __init__(self, mode: str) -> None:
                self.mode = mode

            def predict(self, x_np):
                if self.mode == "raise":
                    raise RuntimeError("simulated OV inference failure")
                if self.mode == "nan":
                    return np.full((1, NUM_ACTIONS), np.nan, dtype=np.float32)
                return np.zeros((1, NUM_ACTIONS - 1), dtype=np.float32)  # wrong class count

        hw = types.SimpleNamespace(ov_devices=["CPU"], has_cuda=False)
        for mode in ("raise", "nan", "shape"):
            with self.subTest(mode=mode):
                tm = dataclasses.replace(MODELS["mlp"], ov_runners={}, active_infer_backend="torch_cpu",
                                         infer_ms=1.234)
                with mock.patch.object(model_module, "detect_hardware", return_value=hw), \
                        mock.patch.object(model_module, "compile_torch_to_openvino", return_value=_Runner(mode)):
                    with self.assertRaises(Exception):
                        prepare_infer_backend(tm, "ov_cpu")
                self.assertEqual(tm.ov_runners, {})
                self.assertEqual(tm.active_infer_backend, "torch_cpu")
                self.assertEqual(tm.infer_ms, 1.234)
                action, probs = predict_action(tm, _obs(1))  # still serves on the verified PyTorch path
                self.assertIn(action, (0, 1, 2))
                self.assertTrue(np.all(np.isfinite(probs)))

    def test_non_finite_parameters_fail_at_load_and_fall_back_to_legacy(self) -> None:
        import copy
        import shutil

        from model import load_checkpoint_with_fallback

        def linear_with(mutate):
            est = copy.deepcopy(MODELS["linear"].estimator)
            mutate(est)
            return {"model_type": "linear", "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS, "estimator": est}

        def set_intercept(est):
            est.named_steps["clf"].intercept_[0] = -np.inf

        def set_coef(est):
            est.named_steps["clf"].coef_[0, 0] = np.inf

        def set_scale(est):
            est.named_steps["scaler"].scale_[0] = np.inf

        def set_mean(est):
            est.named_steps["scaler"].mean_[0] = np.inf

        joblib_cases = {"intercept": set_intercept, "coef": set_coef, "scale": set_scale, "mean": set_mean}
        with tempfile.TemporaryDirectory() as d:
            models_dir = Path(d)
            shutil.copy(checkpoint_path_for(TMP_DIR / "models", "linear"),
                        legacy_checkpoint_path_for(models_dir, "linear"))
            shutil.copy(checkpoint_path_for(TMP_DIR / "models", "mlp"), legacy_checkpoint_path_for(models_dir, "mlp"))
            for name, mutate in joblib_cases.items():
                with self.subTest(case=name):
                    joblib.dump(linear_with(mutate), checkpoint_path_for(models_dir, "linear"))
                    with self.assertRaises(ValueError):
                        load_checkpoint(checkpoint_path_for(models_dir, "linear"))
                    tm, errors = load_checkpoint_with_fallback(models_dir, "linear")
                    self.assertIsNotNone(tm)
                    self.assertEqual(len(errors), 1)
                    self.assertIn(checkpoint_path_for(models_dir, "linear").name, errors[0])
            # A -inf hidden bias is silenced by ReLU on the all-zero probe, yet the weights are corrupt.
            state = {k: v.detach().cpu().clone() for k, v in MODELS["mlp"].estimator.state_dict().items()}
            state["net.0.bias"][0] = float("-inf")
            torch.save({"model_type": "mlp", "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS, "state_dict": state},
                       checkpoint_path_for(models_dir, "mlp"))
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint_path_for(models_dir, "mlp"))
            tm, errors = load_checkpoint_with_fallback(models_dir, "mlp")
            self.assertIsNotNone(tm)
            self.assertEqual(len(errors), 1)

    def test_confusion_chart_uses_order_sensitive_fingerprint(self) -> None:
        import dataclasses

        import matplotlib.pyplot as plt

        from visualizer import build_figure

        rng = np.random.default_rng(31)
        small = DrivingDataset()
        for o, a in zip(*DATASET.to_arrays()):
            if rng.random() < 0.4:
                small.add(o, int(a))
        x, y = small.to_arrays()
        shuffled = DrivingDataset()
        for i in rng.permutation(len(x)):
            shuffled.add(x[i], int(y[i]))
        self.assertEqual(shuffled.class_counts(), small.class_counts())
        self.assertEqual(small.fingerprint(), small.snapshot().fingerprint())
        self.assertNotEqual(shuffled.fingerprint(), small.fingerprint())

        tm, _ = train_model(small, model_type="linear", mirror_augment=False, train_device="cpu")
        self.assertEqual(tm.dataset_fingerprint, small.fingerprint())
        legacy = dataclasses.replace(tm, dataset_fingerprint=None)
        differs = f"현재 데이터셋 분할 평가 (학습 데이터와 다름 · 체크포인트 기록 {tm.val_metric:.1f}%)"
        unverified = f"현재 데이터셋 분할 평가 (체크포인트 기록 {tm.val_metric:.1f}%)"

        def chart_text(ds, model):
            fig = build_figure("classification", ds, {"linear": model}, "linear", [])
            try:
                return "\n".join(_figure_texts(fig))
            finally:
                plt.close(fig)

        exact = chart_text(small, tm)
        self.assertNotIn("현재 데이터셋 분할 평가", exact)
        self.assertIn(f"전체 정확도: {tm.val_metric:.1f}%", exact)
        reordered = chart_text(shuffled, tm)
        self.assertIn(differs, reordered)
        for ds in (small, DATASET):
            with self.subTest(unverified=len(ds)):
                text = chart_text(ds, legacy)
                self.assertIn(unverified, text)
                self.assertNotIn("다름", text)

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "goal_linear.joblib"
            save_checkpoint(tm, path)
            self.assertEqual(load_checkpoint(path, expected_model_type="linear").dataset_fingerprint,
                             small.fingerprint())
            pt = Path(d) / "goal_mlp.pt"
            save_checkpoint(dataclasses.replace(MODELS["mlp"], dataset_fingerprint="ab" * 32), pt)
            self.assertEqual(load_checkpoint(pt).dataset_fingerprint, "ab" * 32)
            old = Path(d) / "old_linear.joblib"
            joblib.dump({"model_type": "linear", "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS,
                         "estimator": tm.estimator}, old)
            self.assertIsNone(load_checkpoint(old).dataset_fingerprint)

    def test_default_named_pipeline_linear_supported(self) -> None:
        import dataclasses

        import matplotlib.pyplot as plt
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from visualizer import build_figure

        script = TestReviewF9FigureLinearCoefficients._script()
        x, _ = DATASET.to_arrays()
        y = np.random.default_rng(13).integers(0, NUM_ACTIONS, len(x))
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(x, y)
        self.assertNotIn("clf", est.named_steps)
        tm = dataclasses.replace(MODELS["linear"], estimator=est)
        fig = build_figure("importance", DATASET, {"linear": tm}, "linear", [])
        try:
            self.assertIn("로지스틱 회귀", fig._suptitle.get_text())
        finally:
            plt.close(fig)
        np.testing.assert_allclose(script.linear_class_coefficients(est), est.steps[-1][1].coef_)


# ----------------------------------------------------------------------
class TestFinalFixCycle4(unittest.TestCase):
    """Fix cycle 4: None-returning OpenVINO runner, nested pipeline parameters, training-array fingerprint."""

    def test_openvino_runner_returning_none_is_an_inference_error(self) -> None:
        import dataclasses
        import types

        import model as model_module

        class _NoneRunner:
            compile_ms = 2.0

            def predict(self, x_np):
                return None

        hw = types.SimpleNamespace(ov_devices=["CPU"], has_cuda=False)
        # 1) Staging a fresh runner that returns None must fail instead of passing via the PyTorch path.
        tm = dataclasses.replace(MODELS["mlp"], ov_runners={}, active_infer_backend="torch_cpu", infer_ms=1.234)
        with mock.patch.object(model_module, "detect_hardware", return_value=hw), \
                mock.patch.object(model_module, "compile_torch_to_openvino", return_value=_NoneRunner()):
            with self.assertRaises(Exception):
                prepare_infer_backend(tm, "ov_cpu")
        self.assertEqual(tm.ov_runners, {})
        self.assertEqual(tm.active_infer_backend, "torch_cpu")
        self.assertEqual(tm.infer_ms, 1.234)

        # 2) An active OpenVINO backend whose runner returns None raises instead of silently using PyTorch.
        active = dataclasses.replace(MODELS["mlp"], ov_runners={"CPU": _NoneRunner()}, active_infer_backend="ov_cpu")
        with self.assertRaises((RuntimeError, ValueError)):
            predict_action(active, _obs(2))

        # 3) Evicting a failing cached runner that was active re-measures latency on the PyTorch fallback.
        stale = dataclasses.replace(MODELS["mlp"], ov_runners={"CPU": _NoneRunner()}, active_infer_backend="ov_cpu",
                                    infer_ms=999.0)
        with mock.patch.object(model_module, "detect_hardware", return_value=hw):
            with self.assertRaises(Exception):
                prepare_infer_backend(stale, "ov_cpu")
        self.assertEqual(stale.ov_runners, {})
        self.assertEqual(stale.active_infer_backend, "torch_cpu")
        self.assertNotEqual(stale.infer_ms, 999.0)
        self.assertTrue(np.isfinite(stale.infer_ms) and 0.0 < stale.infer_ms < 999.0)

    def test_nested_pipeline_non_finite_scaler_fails_at_load(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        x, y = DATASET.to_arrays()

        def nested(mutate):
            est = Pipeline([
                ("outer", Pipeline([("pre", Pipeline([("scaler", StandardScaler())]))])),
                ("clf", LogisticRegression(max_iter=300)),
            ]).fit(x, y)
            mutate(est)
            return est

        def scale(est):
            est.steps[0][1].steps[0][1].steps[0][1].scale_[0] = np.inf

        def mean(est):
            est.steps[0][1].steps[0][1].steps[0][1].mean_[0] = np.nan

        def var(est):
            est.steps[0][1].steps[0][1].steps[0][1].var_[0] = np.inf

        def inner_clf(est):  # a LogisticRegression nested inside a sub-pipeline is checked too
            inner = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=300))]).fit(x, y)
            inner.steps[1][1].intercept_[0] = -np.inf
            est.steps[0] = ("outer", Pipeline([("pre", inner)]))

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "goal_linear.joblib"
            joblib.dump({"model_type": "linear", "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS,
                         "estimator": nested(lambda est: None)}, path)
            self.assertEqual(load_checkpoint(path, expected_model_type="linear").model_type, "linear")
            for name, mutate in {"scale": scale, "mean": mean, "var": var}.items():
                with self.subTest(case=name):
                    joblib.dump({"model_type": "linear", "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS,
                                 "estimator": nested(mutate)}, path)
                    with self.assertRaisesRegex(ValueError, "NaN / Inf"):
                        load_checkpoint(path, expected_model_type="linear")
            from model import _require_finite_parameters
            import dataclasses
            tm = dataclasses.replace(MODELS["linear"], estimator=nested(inner_clf))
            with self.assertRaisesRegex(ValueError, "intercept_"):
                _require_finite_parameters(tm, path)

    def test_fingerprint_comes_from_training_arrays_not_mutated_dataset(self) -> None:
        rng = np.random.default_rng(47)
        ds = DrivingDataset()
        for o, a in zip(*DATASET.to_arrays()):
            if rng.random() < 0.3:
                ds.add(o, int(a))
        x0, y0 = ds.to_arrays()
        original = ds.fingerprint()

        def reorder(_history) -> None:  # mutate the live dataset mid-training: same rows, reversed order
            x, y = ds.to_arrays()
            ds.clear()
            for i in range(len(x) - 1, -1, -1):
                ds.add(x[i], int(y[i]))

        for model_type in ("linear", "mlp"):
            with self.subTest(model_type=model_type):
                ds.clear()
                for o, a in zip(x0, y0):
                    ds.add(o, int(a))
                tm, _ = train_model(ds, model_type=model_type, epochs=1, mirror_augment=False,
                                    train_device="cpu", on_epoch=reorder)
                self.assertNotEqual(ds.fingerprint(), original)  # the live dataset was reordered (odd # calls)
                self.assertEqual(tm.dataset_fingerprint, original)  # but the model records what it trained on

        from dataset import fingerprint_arrays  # the shared helper used by both fingerprint() and train_model()

        self.assertEqual(fingerprint_arrays(x0, y0), original)
        self.assertNotEqual(fingerprint_arrays(x0[::-1], y0[::-1]), original)


# ----------------------------------------------------------------------
class _FakeKeys:
    def __init__(self, *pressed: int) -> None:
        self.pressed = set(pressed)

    def __getitem__(self, key: int) -> bool:
        return key in self.pressed


class TestHeadlessApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import app as app_module

        cls.app_module = app_module
        cls.patches = [
            mock.patch.object(app_module, "MODELS_DIR", TMP_DIR / "models"),
            mock.patch.object(app_module, "DATA_PATH", TMP_DIR / "dataset.npz"),
            mock.patch.object(app_module, "BENCH_CSV_PATH", TMP_DIR / "bench.csv"),
            mock.patch.object(app_module, "PLOTS_DIR", TMP_DIR / "plots"),
        ]
        for p in cls.patches:
            p.start()
        cls.app = app_module.LidarPipelineApp()

    @classmethod
    def tearDownClass(cls) -> None:
        import pygame

        cls._wait_viz()
        for p in cls.patches:
            p.stop()
        pygame.quit()

    @classmethod
    def _wait_viz(cls, timeout: float = 60.0) -> None:
        t0 = time.monotonic()
        while cls.app.is_rendering_viz and time.monotonic() - t0 < timeout:
            time.sleep(0.05)

    def test_01_autoload(self) -> None:
        self.assertEqual(set(self.app.trained_models), set(MODEL_TYPES))
        self.assertEqual(len(self.app.dataset), len(DATASET))

    def test_02_render_all_tabs(self) -> None:
        for tab in ("collect", "train", "test", "viz", "guide"):
            with self.subTest(tab=tab):
                self.app._switch_tab(tab)
                self.app._render()
                self._wait_viz()
                self.app._render()
        self.app._switch_tab("train")
        self.app.train_settings_open = True
        self.app._render()
        keys = {h.key for h in self.app.ui.hits}
        self.assertIn("toggle_balance", keys)
        self.assertFalse(keys & {"task_reg", "task_cls"})
        self.app._switch_tab("viz")
        self._wait_viz()
        self.app._render()
        self.assertNotIn("viz_toggle_color", {h.key for h in self.app.ui.hits})

    def test_03_race_moves_all_contenders(self) -> None:
        app = self.app
        app._switch_tab("test")
        app._on_button("mode_race")
        self.assertEqual(len(app.race_contenders), 1 + len(MODEL_TYPES))
        app._on_button("toggle_test")
        self.assertTrue(app.race_running)
        for _ in range(60):
            app._update(1.0 / 60.0)
        app._render()
        for cont in app.race_contenders:
            self.assertGreater(cont.state.distance, 0.0, cont.key)
        app.race_running = False

    def test_04_single_run_prediction_info(self) -> None:
        app = self.app
        app._switch_tab("test")
        app._on_button("mode_single")
        for key in ("baseline", "mlp", "rf"):
            with self.subTest(model=key):
                app._on_button(f"pick_single_{key}")
                app._on_button("toggle_test")
                self.assertTrue(app.single_running)
                app._update(1.0 / 60.0)
                info = app.last_pred_info
                self.assertEqual(set(info), {"action", "probs"})
                self.assertIn(info["action"], (0, 1, 2))
                self.assertEqual(np.asarray(info["probs"]).shape, (3,))
                self.assertAlmostEqual(float(np.sum(info["probs"])), 1.0, places=5)
                if key == "baseline":
                    self.assertEqual(float(np.max(info["probs"])), 1.0)
                app._render()
                app.single_running = False

    def test_05_expert_live_grows_dataset(self) -> None:
        app = self.app
        app._switch_tab("collect")
        n0 = len(app.dataset)
        app._on_button("toggle_expert_live")
        self.assertTrue(app.expert_live)
        for _ in range(12):
            app._update(0.1)
        app._render()
        self.assertGreater(len(app.dataset), n0)
        app._on_button("toggle_expert_live")
        self.assertFalse(app.expert_live)

    def test_06_manual_recording(self) -> None:
        import pygame

        app = self.app
        app._switch_tab("collect")
        app.sim.reset_robot()
        app._on_button("toggle_record")
        self.assertTrue(app.recording)
        with mock.patch.object(pygame.key, "get_pressed", return_value=_FakeKeys(pygame.K_a)):
            self.assertEqual(app._read_manual_action(), ACTION_LEFT)
            n0, left0 = len(app.dataset), app.dataset.class_counts()[ACTION_LEFT]
            app._update(0.1)
            self.assertEqual(len(app.dataset), n0 + 1)
            self.assertEqual(app.dataset.class_counts()[ACTION_LEFT], left0 + 1)
        with mock.patch.object(pygame.key, "get_pressed", return_value=_FakeKeys()):
            self.assertIsNone(app._read_manual_action())
            n1 = len(app.dataset)
            app._update(0.1)
            self.assertEqual(len(app.dataset), n1)
        for keys, expected in (((pygame.K_a, pygame.K_d), None), ((pygame.K_a, pygame.K_d, pygame.K_w), ACTION_STRAIGHT),
                               ((pygame.K_RIGHT,), ACTION_RIGHT), ((pygame.K_UP,), ACTION_STRAIGHT),
                               ((pygame.K_s,), None), ((pygame.K_DOWN,), None)):
            with mock.patch.object(pygame.key, "get_pressed", return_value=_FakeKeys(*keys)):
                self.assertEqual(app._read_manual_action(), expected, keys)
        app._on_button("toggle_record")
        self.assertFalse(app.recording)

    def test_07_build_all_figures(self) -> None:
        import matplotlib.pyplot as plt

        from visualizer import CHART_CHOICES, build_figure

        self._wait_viz()
        self.assertEqual(len(CHART_CHOICES), 5)
        self.assertNotIn("regression", [c for c, _ in CHART_CHOICES])
        bench = evaluate_models_on_random_maps(MODELS, num_maps=1, seed_start=1000)
        for chart, _ in CHART_CHOICES:
            with self.subTest(chart=chart):
                fig = build_figure(chart, self.app.dataset, self.app.trained_models, "mlp", bench)
                self.assertIsNotNone(fig)
                plt.close(fig)
        for m in ("linear", "rf", "knn", "cnn1d"):
            fig = build_figure("importance", self.app.dataset, self.app.trained_models, m, bench)
            plt.close(fig)
        fig = build_figure("importance", self.app.dataset, {}, "mlp", bench)  # ANOVA fallback
        plt.close(fig)

    # --- Review regressions ------------------------------------------------
    @classmethod
    def _wait_until(cls, predicate, timeout: float = 30.0) -> None:
        t0 = time.monotonic()
        while not predicate() and time.monotonic() - t0 < timeout:
            time.sleep(0.02)

    @staticmethod
    def _clone_dataset(source: DrivingDataset) -> DrivingDataset:
        clone = DrivingDataset()
        for o, a in zip(*source.to_arrays()):
            clone.add(o, int(a))
        return clone

    def test_08_benchmark_blocks_race_and_single_start(self) -> None:
        import pygame

        app = self.app
        app._switch_tab("test")
        app.race_running = app.single_running = False
        app.is_benchmarking = True
        try:
            for mode in ("race", "single"):
                with self.subTest(mode=mode):
                    app._on_button(f"mode_{mode}")
                    app._on_button("toggle_test")
                    app._handle_keydown(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
                    self.assertFalse(app.race_running)
                    self.assertFalse(app.single_running)
                    app._render()
                    hits = [h for h in app.ui.hits if h.key == "toggle_test"]
                    self.assertEqual(len(hits), 1)
                    self.assertTrue(hits[0].disabled)
                    self.assertIsNone(app.ui.hit_test(hits[0].rect.center))
        finally:
            app.is_benchmarking = False
            app._on_button("mode_race")
        app._render()
        self.assertFalse([h for h in app.ui.hits if h.key == "toggle_test"][0].disabled)

    def test_09_training_uses_dataset_snapshot_and_keeps_active_race(self) -> None:
        app_module, app = self.app_module, self.app
        orig_dataset, orig_models = app.dataset, dict(app.trained_models)
        app.dataset = self._clone_dataset(orig_dataset)
        n0 = len(app.dataset)
        gate, seen = threading.Event(), {}

        def fake_train(dataset, model_type, **kwargs):
            gate.wait(10)
            seen["len"], seen["same"] = len(dataset), dataset is app.dataset
            seen["thread"] = threading.current_thread()
            return MODELS["linear"], TrainHistory(model_type="linear")

        try:
            app._switch_tab("test")
            app._on_button("mode_race")
            app.selected_model_type = "linear"
            with mock.patch.object(app_module, "train_model", fake_train):
                app._start_training(train_all=False)
                self.assertTrue(app.is_training)
                app._on_button("clear_data")
                self.assertEqual(len(app.dataset), 0)
                app._on_button("toggle_test")
                self.assertTrue(app.race_running)
                for _ in range(10):
                    app._update(1.0 / 30.0)
                before = [(id(c), c.state.elapsed) for c in app.race_contenders]
                gate.set()
                self._wait_until(lambda: "thread" in seen)
                seen.pop("thread").join(10)
            self.assertEqual(seen, {"len": n0, "same": False})
            self.assertTrue(app.race_running)
            self.assertEqual([i for i, _ in before], [id(c) for c in app.race_contenders])
            self.assertEqual([e for _, e in before], [c.state.elapsed for c in app.race_contenders])
            # The finished job is applied on the next UI tick; the running race keeps its robots and clocks.
            app._update(1.0 / 30.0)
            self.assertFalse(app.is_training)
            self.assertTrue(app.race_running)
            self.assertEqual([i for i, _ in before], [id(c) for c in app.race_contenders])
            self.assertGreater(app.race_contenders[0].state.elapsed, before[0][1])
            self.assertTrue(app.race_rebuild_pending)
            # Restarting picks up the freshly trained model.
            app._on_button("toggle_test")
            app._on_button("toggle_test")
            self.assertIs(next(c.model for c in app.race_contenders if c.key == "linear"), MODELS["linear"])
            self.assertTrue(all(c.state.elapsed == 0.0 for c in app.race_contenders))
        finally:
            gate.set()
            app.race_running = False
            app.dataset = orig_dataset
            app.trained_models.clear()
            app.trained_models.update(orig_models)
            app._reset_race()

    def test_10_viz_jobs_use_dataset_snapshot(self) -> None:
        import matplotlib.pyplot as plt

        app_module, app = self.app_module, self.app
        self._wait_viz()
        orig_dataset = app.dataset
        n0 = len(orig_dataset)
        try:
            for job in ("refresh", "export"):
                with self.subTest(job=job):
                    app.dataset = self._clone_dataset(orig_dataset)
                    gate, seen = threading.Event(), []

                    def fake_build(chart_type, dataset, **kwargs):
                        gate.wait(10)
                        seen.append((len(dataset), dataset is app.dataset))
                        return plt.figure(figsize=(2, 2))

                    with mock.patch.object(app_module, "build_figure", fake_build), \
                            mock.patch.object(app_module, "save_figure_png", lambda fig, path, dpi=160: plt.close(fig)):
                        if job == "refresh":
                            app._refresh_viz()
                        else:
                            app._export_viz_png(export_all=False)
                        self.assertTrue(app.is_rendering_viz)
                        app.dataset.clear()
                        gate.set()
                        self._wait_viz()
                    self.assertEqual(seen, [(n0, False)])
        finally:
            app.dataset = orig_dataset

    def test_11_invalid_observation_keeps_robot_stationary(self) -> None:
        app = self.app
        app._switch_tab("test")
        app._on_button("mode_race")
        app._on_button("toggle_test")
        self.assertTrue(app.race_running)
        try:
            for cont in app.race_contenders:
                cont.state.last_obs = np.full(OBS_DIM, np.nan, dtype=np.float32)
            app._update(1.0 / 30.0)
            for cont in app.race_contenders:
                self.assertEqual(cont.state.distance, 0.0, cont.key)
                self.assertFalse(cont.state.collided, cont.key)
        finally:
            app.race_running = False
        app._on_button("mode_single")
        app._on_button("pick_single_mlp")
        app._on_button("toggle_test")
        try:
            app.sim.robot.last_obs = np.full(OBS_DIM, np.nan, dtype=np.float32)
            d0 = app.sim.robot.distance
            app._update(1.0 / 30.0)
            self.assertEqual(app.sim.robot.distance, d0)
            app._render()
        finally:
            app.single_running = False
            app._on_button("mode_race")

    def test_12_autoload_falls_back_to_legacy_and_reports_errors(self) -> None:
        import shutil

        app_module, app = self.app_module, self.app
        saved = (dict(app.trained_models), app.active_model, app.selected_model_type, app.single_model_key)
        with tempfile.TemporaryDirectory() as d:
            models_dir = Path(d)
            checkpoint_path_for(models_dir, "rf").write_bytes(b"not a checkpoint")
            shutil.copy(checkpoint_path_for(TMP_DIR / "models", "rf"), legacy_checkpoint_path_for(models_dir, "rf"))
            joblib.dump({"model_type": "linear", "obs_dim": OBS_DIM, "num_beams": NUM_BEAMS,
                         "estimator": MODELS["rf"].estimator}, checkpoint_path_for(models_dir, "linear"))
            shutil.copy(checkpoint_path_for(TMP_DIR / "models", "mlp"), checkpoint_path_for(models_dir, "mlp"))
            try:
                with mock.patch.object(app_module, "MODELS_DIR", models_dir):
                    app.trained_models.clear()
                    app.active_model = None
                    app._autoload_saved_models()
                self.assertEqual(set(app.trained_models), {"rf", "mlp"})
                self.assertEqual(app.trained_models["rf"].model_type, "rf")
                errors = "\n".join(app.model_load_errors)
                self.assertIn("goal_rf.joblib", errors)
                self.assertIn("goal_linear.joblib", errors)
                self.assertNotIn("goal_mlp.pt", errors)
                self.assertIn("goal_rf.joblib", app.status_msg)
            finally:
                app.trained_models.clear()
                app.trained_models.update(saved[0])
                app.active_model, app.selected_model_type, app.single_model_key = saved[1:]
                app._reset_race()

    def test_13_backend_switch_failure_rolls_back(self) -> None:
        app_module, app = self.app_module, self.app
        torch_models = [tm for tm in app.trained_models.values() if tm.backend == "torch"]
        self.assertGreaterEqual(len(torch_models), 2)
        prev_backend = app.infer_backend
        prev = {id(tm): tm.active_infer_backend for tm in torch_models}
        calls: list = []

        def flaky_prepare(tm, backend, export_dir=None):
            calls.append((tm.model_type, backend))
            if backend != prev[id(tm)] and len(calls) == 2:
                raise RuntimeError("simulated compile failure")
            tm.active_infer_backend = backend
            return 0.0

        with mock.patch.object(app_module, "prepare_infer_backend", flaky_prepare):
            app._switch_infer_backend("ov_npu")
            self._wait_until(lambda: not app.is_compiling_hw)
        self.assertFalse(app.is_compiling_hw)
        self.assertEqual(app.infer_backend, prev_backend)
        for tm in torch_models:
            self.assertEqual(tm.active_infer_backend, prev[id(tm)], tm.model_type)
        self.assertIn("simulated compile failure", app.status_msg)

    # --- Sign-off review regressions --------------------------------------
    def _run_gated_training(self, fresh, model_type: str, before_release=None):
        """Start a single-model training whose fake ``train_model`` blocks until released; return the worker."""
        app_module, app = self.app_module, self.app
        gate, seen = threading.Event(), {}

        def fake_train(dataset, model_type, **kwargs):
            seen["thread"] = threading.current_thread()
            gate.wait(10)
            return fresh, TrainHistory(model_type=fresh.model_type)

        app.selected_model_type = model_type
        with mock.patch.object(app_module, "train_model", fake_train):
            app._start_training(train_all=False)
            self.assertTrue(app.is_training)
            self._wait_until(lambda: "thread" in seen)
            if before_release is not None:
                before_release()
            gate.set()
            seen["thread"].join(10)
        self.assertFalse(seen["thread"].is_alive())
        return seen["thread"]

    def _snapshot_models(self):
        app = self.app
        return dict(app.trained_models), app.active_model, app.selected_model_type, app.single_model_key

    def _restore_models(self, saved) -> None:
        app = self.app
        app.race_running = app.single_running = False
        app.trained_models.clear()
        app.trained_models.update(saved[0])
        app.active_model, app.selected_model_type, app.single_model_key = saved[1:]
        app._reset_race()

    def test_14_training_completion_applied_on_ui_thread_without_hijacking_single_run(self) -> None:
        import dataclasses

        app = self.app
        saved = self._snapshot_models()
        dt = 1.0 / 30.0
        try:
            app._switch_tab("test")
            app._on_button("mode_single")
            app._on_button("pick_single_baseline")
            app._reset_current_scene()
            fresh = dataclasses.replace(MODELS["linear"])
            state = {}

            def start_baseline_run() -> None:
                app._on_button("toggle_test")
                self.assertTrue(app.single_running)
                app._update(dt)
                state["elapsed"], state["status"] = app.sim.robot.elapsed, app.status_msg

            self._run_gated_training(fresh, "linear", start_baseline_run)
            # The worker thread finished but must not have touched UI / model / single-run state.
            self.assertIsNot(app.trained_models.get("linear"), fresh)
            self.assertEqual(app.single_model_key, "baseline")
            self.assertEqual(app.status_msg, state["status"])
            self.assertEqual(app.sim.robot.elapsed, state["elapsed"])
            with mock.patch.object(app, "_policy_action", wraps=app._policy_action) as spy:
                app._update(dt)
            self.assertIs(app.trained_models["linear"], fresh)
            self.assertFalse(app.is_training)
            self.assertIn("학습 완료", app.status_msg)
            self.assertEqual(app.single_model_key, "baseline")
            self.assertTrue(app.single_running)
            self.assertIsNone(spy.call_args[0][0])  # still the baseline driver
            self.assertAlmostEqual(app.sim.robot.elapsed, state["elapsed"] + dt, places=6)
            app.single_running = False

            # An active single run on a model keeps that model object even when it is retrained.
            app._reset_current_scene()
            app._on_button("pick_single_linear")
            old = app.trained_models["linear"]
            fresh2 = dataclasses.replace(MODELS["linear"])

            def start_linear_run() -> None:
                app._on_button("toggle_test")
                self.assertTrue(app.single_running)

            self._run_gated_training(fresh2, "linear", start_linear_run)
            with mock.patch.object(app, "_policy_action", wraps=app._policy_action) as spy:
                app._update(dt)
            self.assertIs(app.trained_models["linear"], fresh2)
            self.assertIs(spy.call_args[0][0], old)
            # Restarting the run picks up the retrained model.
            app.single_running = False
            app._on_button("toggle_test")
            with mock.patch.object(app, "_policy_action", wraps=app._policy_action) as spy:
                app._update(dt)
            self.assertIs(spy.call_args[0][0], fresh2)
        finally:
            self._restore_models(saved)
            app._on_button("mode_race")

    def test_15_invalid_collect_frames_are_handled(self) -> None:
        import pygame

        app = self.app
        app._switch_tab("collect")
        app.sim.reset_robot()
        bad_frames = {"nan": np.full(OBS_DIM, np.nan, dtype=np.float32),
                      "complex": np.ones(OBS_DIM, dtype=np.complex64),
                      "short": np.zeros(OBS_DIM - 1, dtype=np.float32)}
        for mode in ("expert", "manual"):
            for name, bad in bad_frames.items():
                with self.subTest(mode=mode, frame=name):
                    app._on_button("toggle_expert_live" if mode == "expert" else "toggle_record")
                    self.assertTrue(app.expert_live if mode == "expert" else app.recording)
                    robot = app.sim.robot
                    pose0, n0 = (robot.x, robot.y, robot.theta, robot.distance), len(app.dataset)
                    try:
                        with mock.patch.object(app.sim, "get_observation", return_value=bad), \
                                mock.patch.object(pygame.key, "get_pressed", return_value=_FakeKeys(pygame.K_w)):
                            for _ in range(3):
                                app._update(0.1)
                        robot = app.sim.robot
                        self.assertEqual((robot.x, robot.y, robot.theta, robot.distance), pose0)
                        self.assertEqual(len(app.dataset), n0)
                        self.assertIn("관측", app.status_msg)
                    finally:
                        app._on_button("toggle_expert_live" if mode == "expert" else "toggle_record")
        self.assertFalse(app.expert_live or app.recording)

    def test_16_autoload_enforces_requested_model_identity(self) -> None:
        import shutil

        app_module, app = self.app_module, self.app
        saved = self._snapshot_models()
        with tempfile.TemporaryDirectory() as d:
            models_dir = Path(d)
            # Canonical knn file holds a perfectly valid *linear* checkpoint; the legacy sibling is a real KNN.
            shutil.copy(checkpoint_path_for(TMP_DIR / "models", "linear"), checkpoint_path_for(models_dir, "knn"))
            shutil.copy(checkpoint_path_for(TMP_DIR / "models", "knn"), legacy_checkpoint_path_for(models_dir, "knn"))
            # Canonical cnn1d file holds a valid MLP checkpoint and has no fallback.
            shutil.copy(checkpoint_path_for(TMP_DIR / "models", "mlp"), checkpoint_path_for(models_dir, "cnn1d"))
            try:
                with mock.patch.object(app_module, "MODELS_DIR", models_dir):
                    app.trained_models.clear()
                    app.active_model = None
                    app._autoload_saved_models()
                self.assertEqual(set(app.trained_models), {"knn"})
                self.assertEqual(app.trained_models["knn"].model_type, "knn")
                errors = "\n".join(app.model_load_errors)
                self.assertIn("goal_knn.joblib", errors)
                self.assertIn("goal_cnn1d.pt", errors)
            finally:
                self._restore_models(saved)

    def test_17_benchmark_and_training_blocked_during_backend_switch(self) -> None:
        app_module, app = self.app_module, self.app
        calls: list = []
        app.is_compiling_hw = True
        try:
            self.assertTrue(app.infer_switching)
            with mock.patch.object(app_module, "train_model", lambda *a, **k: calls.append("train")), \
                    mock.patch.object(app_module, "evaluate_models_on_random_maps",
                                      lambda *a, **k: calls.append("bench") or []):
                app._start_20map_benchmark()
                app._on_button("run_bench")
                self.assertFalse(app.is_benchmarking)
                for train_all in (False, True):
                    app._start_training(train_all=train_all)
                app._on_button("start_train")
                app._on_button("train_all")
                self.assertFalse(app.is_training)
                time.sleep(0.05)
            self.assertEqual(calls, [])
            for tab, keys in (("test", ("run_bench",)), ("train", ("train_all", "start_train"))):
                app._switch_tab(tab)
                app._render()
                for key in keys:
                    with self.subTest(tab=tab, key=key):
                        hits = [h for h in app.ui.hits if h.key == key]
                        self.assertEqual(len(hits), 1)
                        self.assertTrue(hits[0].disabled)
        finally:
            app.is_compiling_hw = False
        self.assertFalse(app.infer_switching)
        app._render()
        self.assertFalse([h for h in app.ui.hits if h.key == "start_train"][0].disabled)

    def test_18_corrupt_dataset_file_never_crashes_app(self) -> None:
        app_module, app = self.app_module, self.app
        with tempfile.TemporaryDirectory() as d:
            corrupt = Path(d) / "dataset.npz"
            np.savez(corrupt, obs=np.array([[0.1] * OBS_DIM] * 3, dtype=object), actions=np.array([0, 1, 2]))
            n0 = len(app.dataset)
            with mock.patch.object(app_module, "DATA_PATH", corrupt):
                app._on_button("load_data")
                self.assertEqual(len(app.dataset), n0)
                self.assertIn("호환되는 저장 데이터가 없습니다", app.status_msg)
                fresh_app = app_module.LidarPipelineApp()
            self.assertEqual(len(fresh_app.dataset), 0)
            self.assertEqual(set(fresh_app.trained_models), set(MODEL_TYPES))

    def test_19_post_training_backend_failure_reports_actual_backend(self) -> None:
        import dataclasses

        from accelerator import BACKEND_LABELS

        app_module, app = self.app_module, self.app
        saved = self._snapshot_models()
        saved_backend = app.infer_backend
        fresh = dataclasses.replace(MODELS["mlp"], ov_runners={}, active_infer_backend="torch_cpu")

        def failing_prepare(tm, backend, export_dir=None):
            raise RuntimeError("simulated OV compile failure")

        try:
            app._switch_tab("train")
            app.infer_backend = "ov_cpu"
            with mock.patch.object(app_module, "prepare_infer_backend", failing_prepare):
                self._run_gated_training(fresh, "mlp")
                app._update(1.0 / 30.0)
            self.assertFalse(app.is_training)
            self.assertIs(app.trained_models["mlp"], fresh)
            self.assertEqual(fresh.active_infer_backend, "torch_cpu")
            self.assertIn(BACKEND_LABELS["torch_cpu"], app.status_msg)
            self.assertIn("simulated OV compile failure", app.status_msg)
            # No torch model runs on OV CPU any more, so the selected backend follows what actually executes.
            torch_backends = {tm.active_infer_backend for tm in app.trained_models.values() if tm.backend == "torch"}
            self.assertNotIn("ov_cpu", torch_backends)
            self.assertEqual(app.infer_backend, "torch_cpu")
        finally:
            app.infer_backend = saved_backend
            self._restore_models(saved)

    def test_20_map_ab_race_outcomes_are_deterministic(self) -> None:
        app = self.app
        # Use local user checkpoints when all 5 exist; otherwise fall back to the test fixture models so a fresh
        # clone or a user-retrained models/ directory never breaks the suite.
        local_paths = {m: find_checkpoint(ROOT / "models", m) for m in MODEL_TYPES}
        if all(p is not None for p in local_paths.values()):
            contender_models = {m: load_checkpoint(p, expected_model_type=m) for m, p in local_paths.items()}
        else:
            contender_models = dict(app.trained_models)
        expected_baseline = {0: "goal", 1: "crash"}
        saved = dict(app.trained_models)
        try:
            app.trained_models.clear()
            app.trained_models.update(contender_models)
            app._switch_tab("test")
            app._on_button("mode_race")
            for preset, base_outcome in expected_baseline.items():
                with self.subTest(map=preset):
                    runs = []
                    for _ in range(2):
                        app._on_button(f"preset_{preset}")
                        app._on_button("toggle_test")
                        self.assertEqual(len(app.race_contenders), 1 + len(contender_models))
                        for _ in range(int(self.app_module.RACE_TIMEOUT_SEC * 30) + 5):
                            if not app.race_running:
                                break
                            app._update(1.0 / 30.0)
                        self.assertFalse(app.race_running)
                        got = {
                            c.key: (
                                "goal" if c.state.reached_goal else "crash" if c.state.collided else "timeout",
                                round(c.state.elapsed, 4),
                            )
                            for c in app.race_contenders
                        }
                        self.assertEqual(got["baseline"][0], base_outcome)
                        self.assertIn("레이스 종료", app.status_msg)
                        runs.append(got)
                    # Two identical runs on the same preset map must match deterministically.
                    self.assertEqual(runs[0], runs[1])
        finally:
            app.race_running = False
            app.trained_models.clear()
            app.trained_models.update(saved)
            app._reset_race()

    def test_21_transient_invalid_collect_frame_recovers(self) -> None:
        app = self.app
        orig_dataset = app.dataset
        app.dataset = orig_dataset.snapshot()
        try:
            app._switch_tab("collect")
            app.sim.reset_robot()
            app._on_button("toggle_expert_live")
            self.assertTrue(app.expert_live)
            robot = app.sim.robot
            robot.last_obs = np.full(OBS_DIM, np.nan, dtype=np.float32)  # one transient bad frame
            d0, n0 = robot.distance, len(app.dataset)
            app._update(1.0 / 30.0)
            self.assertEqual(app.sim.robot.distance, d0)
            self.assertEqual(len(app.dataset), n0)
            for _ in range(20):
                app._update(1.0 / 15.0)
            self.assertGreater(app.sim.robot.distance + app.sim.robot.elapsed, d0)
            self.assertGreater(len(app.dataset), n0)
        finally:
            if app.expert_live:
                app._on_button("toggle_expert_live")
            app.dataset = orig_dataset

    def test_22_non_finite_beam_distance_does_not_crash_render(self) -> None:
        import pygame

        app = self.app
        app._switch_tab("collect")
        app.sim.reset_robot()
        surf = pygame.Surface((1680, 960))
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad):
                robot = app.sim.robot
                robot.last_distances = np.asarray(robot.last_distances, dtype=np.float32).copy()
                robot.last_distances[3] = bad
                app._render_single_robot(surf, robot)
                app._render()
        app.sim.reset_robot()

    def test_23_ov_first_inference_failure_reports_verified_backend(self) -> None:
        import dataclasses
        import types

        import model as model_module
        from accelerator import BACKEND_LABELS

        class _BrokenRunner:
            compile_ms = 3.0

            def predict(self, x_np):
                raise RuntimeError("simulated OV first inference failure")

        app = self.app
        saved = self._snapshot_models()
        saved_backend = app.infer_backend
        fresh = dataclasses.replace(MODELS["mlp"], ov_runners={}, active_infer_backend="torch_cpu")
        hw = types.SimpleNamespace(ov_devices=["CPU"], has_cuda=False)
        try:
            app._switch_tab("train")
            app.infer_backend = "ov_cpu"
            with mock.patch.object(model_module, "detect_hardware", return_value=hw), \
                    mock.patch.object(model_module, "compile_torch_to_openvino", return_value=_BrokenRunner()):
                self._run_gated_training(fresh, "mlp")
                app._update(1.0 / 30.0)
            self.assertFalse(app.is_training)
            self.assertIs(app.trained_models["mlp"], fresh)
            self.assertEqual(fresh.ov_runners, {})
            self.assertEqual(fresh.active_infer_backend, "torch_cpu")
            self.assertIn(f"[{BACKEND_LABELS['torch_cpu']}]로 실행", app.status_msg)
            self.assertIn("simulated OV first inference failure", app.status_msg)
            self.assertEqual(app.infer_backend, "torch_cpu")
        finally:
            app.infer_backend = saved_backend
            self._restore_models(saved)

    def test_24_late_completion_does_not_revert_live_chart(self) -> None:
        app = self.app
        saved = self._snapshot_models()
        saved_training = (app.is_training, app.train_current_model, app.train_history)
        knn_hist = TrainHistory(model_type="knn", total_epochs=1)
        linear_hist = TrainHistory(model_type="linear", total_epochs=1)
        try:
            app.is_training = True
            with app.train_lock:
                app.train_current_model = "knn"  # the worker already moved on to KNN
                app.train_history = knn_hist
            app._post_training_update("model", model_type="linear", trained=MODELS["linear"],
                                      history=linear_hist, requested_backend=app.infer_backend,
                                      backend_error=None)
            app._apply_training_updates()
            self.assertIs(app.train_history, knn_hist)
            self.assertIs(app.trained_models["linear"], MODELS["linear"])
            # Once training is over, a queued completion still lands on the chart.
            app.is_training = False
            app._post_training_update("model", model_type="linear", trained=MODELS["linear"],
                                      history=linear_hist, requested_backend=app.infer_backend,
                                      backend_error=None)
            app._apply_training_updates()
            self.assertIs(app.train_history, linear_hist)
        finally:
            app.is_training, app.train_current_model, app.train_history = saved_training
            with app.train_lock:
                app.train_done_set.discard("linear")
            self._restore_models(saved)

if __name__ == "__main__":
    unittest.main()
