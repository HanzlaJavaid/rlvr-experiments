from pathlib import Path
import sys
import tempfile
import unittest

from science_gspo_g.config import batch_plan, load_config, protocol_config, runtime_config
from science_gspo_g.execution import call_on_main, choose_checkpoint, require_same_training_world_size, torchrun_command
from science_gspo_g.io import atomic_json, digest, read_jsonl, append_jsonl


ROOT = Path(__file__).parents[1]


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs" / "experiment_g_4x5090.yaml")

    def test_gpu_count_preserves_global_batch_generation_and_protocol(self):
        fingerprint = digest(protocol_config(self.cfg))
        for workers, expected_accumulation in [(2, 32), (4, 16), (8, 8), (16, 4)]:
            with self.subTest(workers=workers):
                result = runtime_config(self.cfg, workers)
                plan = result["runtime"]
                self.assertEqual(plan["gradient_accumulation_steps"], expected_accumulation)
                self.assertEqual(workers * expected_accumulation * self.cfg["training"]["microbatch_per_gpu"], 64)
                self.assertEqual(plan["generation_batch_size"], 64)
                self.assertEqual(plan["questions_per_update"], 16)
                self.assertEqual(digest(protocol_config(result)), fingerprint)
                self.assertLessEqual(plan["judge_workers_per_rank"] * workers, 16)

    def test_invalid_topology_fails_before_gpu_work(self):
        with self.assertRaises(ValueError):
            batch_plan(self.cfg, 3)
        with self.assertRaises(ValueError):
            batch_plan(self.cfg, 32)
        with self.assertRaises(ValueError):
            torchrun_command("config.yaml", gpus_per_node=8, nodes=2, master_addr="127.0.0.1")

    def test_single_machine_launch_and_multi_machine_launch_are_explicit(self):
        single = torchrun_command("config.yaml", gpus_per_node=4)
        self.assertEqual(single[:3], [sys.executable, "-m", "torch.distributed.run"])
        self.assertIn("--standalone", single)
        multi = torchrun_command("config.yaml", gpus_per_node=8, nodes=2, node_rank=1, master_addr="10.0.0.7")
        self.assertNotIn("--standalone", multi)
        for argument in ("--nnodes=2", "--node_rank=1", "--nproc_per_node=8", "--master_addr=10.0.0.7"):
            self.assertIn(argument, multi)

    def test_checkpoint_recovery_ignores_latest_incomplete_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "checkpoint-25"
            good.mkdir()
            required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json",
                        "adapter_model.safetensors", "rng_state_0.pth", "rng_state_1.pth"]
            for name in required:
                (good / name).write_text("fixture")
            atomic_json(good / "checkpoint_complete.json", {"protocol_hash": "p", "world_size": 2})
            bad = root / "checkpoint-50"
            bad.mkdir()
            atomic_json(bad / "checkpoint_complete.json", {"protocol_hash": "p", "world_size": 2})
            self.assertEqual(choose_checkpoint(root, "auto", protocol_hash="p", world_size=2), str(good))
            with self.assertRaises(ValueError):
                choose_checkpoint(root, str(bad), protocol_hash="p", world_size=2)
            with self.assertRaises(ValueError):
                choose_checkpoint(root, "auto", protocol_hash="p", world_size=4)

    def test_world_size_change_is_allowed_for_final_evaluation_but_not_training_resume(self):
        with self.assertRaises(ValueError):
            require_same_training_world_size({"training_world_size": 8}, 4, training_finished=False)
        require_same_training_world_size({"training_world_size": 8}, 4, training_finished=True)

    def test_jsonl_tail_recovery_preserves_valid_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_bytes(b'{"x":1}\n{"unfinished":')
            before = path.read_bytes()
            self.assertEqual(read_jsonl(path, repair_tail=False), [{"x": 1}])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(read_jsonl(path), [{"x": 1}])
            append_jsonl(path, [{"x": 2}])
            self.assertEqual(read_jsonl(path), [{"x": 1}, {"x": 2}])

    def test_main_rank_failure_is_communicated_to_other_workers(self):
        sent = []
        def fail():
            raise ValueError("Invalid repository configuration")
        with self.assertRaisesRegex(RuntimeError, "Invalid repository"):
            call_on_main(fail, rank=0, broadcast=lambda values: sent.extend(values))
        def receive(values):
            values[:] = sent
        with self.assertRaisesRegex(RuntimeError, "Invalid repository"):
            call_on_main(lambda: self.fail("A non-main rank must not execute the operation"), rank=1, broadcast=receive)


if __name__ == "__main__":
    unittest.main()

