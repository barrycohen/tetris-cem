import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from numba import njit

SHAPE_DATA = np.zeros((7, 4, 4, 2), dtype=np.int32)
RAW_SHAPES = {
    "I": [[(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 0), (1, 0), (2, 0), (3, 0)]],
    "O": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
    "T": [
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    "S": [[(0, 1), (0, 2), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1), (2, 1)]],
    "Z": [[(0, 0), (0, 1), (1, 1), (1, 2)], [(0, 1), (1, 0), (1, 1), (2, 0)]],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}
PIECE_LIST = ["I", "O", "T", "S", "Z", "J", "L"]
NUM_ROTATIONS = np.array([len(RAW_SHAPES[p]) for p in PIECE_LIST], dtype=np.int32)

for i, p in enumerate(PIECE_LIST):
    shapes = RAW_SHAPES[p]
    for r, s in enumerate(shapes):
        SHAPE_DATA[i, r] = s
    for r in range(len(shapes), 4):  # Pad to avoid index errors
        SHAPE_DATA[i, r] = shapes[-1]


@njit(fastmath=True, cache=True)
def is_valid(board, piece_idx, rot_idx, row, col):
    for b in range(4):
        r_off, c_off = (
            SHAPE_DATA[piece_idx, rot_idx, b, 0],
            SHAPE_DATA[piece_idx, rot_idx, b, 1],
        )
        nr, nc = row + r_off, col + c_off
        if nc < 0 or nc >= 10 or nr >= 20:
            return False
        if nr >= 0 and board[nr, nc] > 0:
            return False
    return True


@njit(fastmath=True, cache=True)
def get_moves(board, piece_idx):
    out_boards = np.empty((64, 20, 10), dtype=np.float32)
    out_lines = np.empty(64, dtype=np.float32)
    count = 0

    for rot in range(NUM_ROTATIONS[piece_idx]):
        for col in range(-2, 10):
            if count >= 64 or not is_valid(board, piece_idx, rot, 0, col):
                continue
            row = 0
            while is_valid(board, piece_idx, rot, row + 1, col):
                row += 1

            # Copy board and place piece
            for r in range(20):
                for c in range(10):
                    out_boards[count, r, c] = board[r, c]
            valid = True
            for b in range(4):
                r_off, c_off = (
                    SHAPE_DATA[piece_idx, rot, b, 0],
                    SHAPE_DATA[piece_idx, rot, b, 1],
                )
                br, bc = row + r_off, col + c_off
                if br < 0:
                    valid = False
                    break
                out_boards[count, br, bc] = 1.0
            if not valid:
                continue

            # Clear lines
            lines, write_row = 0, 19
            for read_row in range(19, -1, -1):
                row_sum = 0.0
                for c in range(10):
                    row_sum += out_boards[count, read_row, c]
                if row_sum >= 10.0:
                    lines += 1
                else:
                    if write_row != read_row:
                        for c in range(10):
                            out_boards[count, write_row, c] = out_boards[
                                count, read_row, c
                            ]
                    write_row -= 1
            for r in range(lines):
                for c in range(10):
                    out_boards[count, r, c] = 0.0

            out_lines[count] = lines
            count += 1
    return count, out_boards, out_lines


class BoardEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 2))
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.fc2(F.relu(self.fc1(self.pool(x).view(x.size(0), -1)))).squeeze(-1)

    def set_params(self, flat_params):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            if idx + n > len(flat_params):
                break
            p.data.copy_(torch.from_numpy(flat_params[idx : idx + n]).view_as(p))
            idx += n


class CEMTrainer:
    def __init__(self, device, pop_size=200, n_elite=20, n_games=5, max_lines=5000):
        self.device = device
        self.pop_size = pop_size
        self.n_elite = n_elite
        self.n_games = n_games
        self.max_lines = max_lines
        self.net = BoardEvaluator().to(device)
        self.net.eval()
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.n_params = sum(p.numel() for p in self.net.parameters())
        self.mean = np.zeros(self.n_params, dtype=np.float32)
        self.std = np.ones(self.n_params, dtype=np.float32)
        self.best_mean = -np.inf
        self.gen = 0

    def play_parallel_games(self):
        n = self.n_games
        boards = np.zeros((n, 20, 10), dtype=np.float32)
        total_lines = np.zeros(n, dtype=np.float32)
        active, pieces = np.ones(n, dtype=bool), np.random.randint(0, 7, size=n)

        while np.any(active):
            batch_boards, batch_lines, move_counts = [], [], []
            for i in range(n):
                if not active[i]:
                    move_counts.append(0)
                    continue
                cnt, opts, lns = get_moves(boards[i], int(pieces[i]))
                if cnt == 0:
                    active[i] = False
                    move_counts.append(0)
                else:
                    batch_boards.append(opts[:cnt])
                    batch_lines.append(lns[:cnt])
                    move_counts.append(cnt)
            if not batch_boards:
                break

            full_input = np.concatenate(batch_boards)
            full_lines = np.concatenate(batch_lines)
            with torch.no_grad():
                scores = self.net(
                    torch.as_tensor(full_input, device=self.device).unsqueeze(1)
                )
            scores += torch.as_tensor(full_lines, device=self.device) * 10.0
            cpu_scores = scores.cpu().numpy()

            cursor = 0
            for i in range(n):
                if move_counts[i] == 0:
                    continue
                best_idx = np.argmax(cpu_scores[cursor : cursor + move_counts[i]])
                boards[i] = full_input[cursor + best_idx]
                total_lines[i] += full_lines[cursor + best_idx]
                active[i] = total_lines[i] < self.max_lines
                if active[i]:
                    pieces[i] = random.randint(0, 6)
                cursor += move_counts[i]
        return total_lines.sum() / n

    def evaluate(self, params):
        self.net.set_params(params)
        return self.play_parallel_games()

    def train_gen(self):
        self.gen += 1
        pop = (
            self.mean
            + np.random.randn(self.pop_size, self.n_params).astype(np.float32)
            * self.std
        )
        scores, t0, last_print = [], time.time(), time.time()

        for i, p in enumerate(pop):
            scores.append(self.evaluate(p))
            if time.time() - last_print >= 60:
                print(f"  Eval: {i + 1}/{self.pop_size}", end="\r")
                last_print = time.time()

        scores = np.array(scores)
        elite = pop[np.argsort(scores)[-self.n_elite :]]
        self.mean = elite.mean(axis=0)
        self.std = np.clip(elite.std(axis=0) * 0.98 + 0.001, 0.02, 2.0)
        improved = scores.mean() > self.best_mean
        if improved:
            self.best_mean = scores.mean()
        return scores, improved, time.time() - t0

    def save(self, path):
        np.savez(
            path, mean=self.mean, std=self.std, best_mean=self.best_mean, gen=self.gen
        )

    def load(self, path="tetris_cem_best.npz"):
        if not os.path.exists(path):
            return False
        try:
            d = np.load(path)
            if d["mean"].shape[0] != self.n_params:
                print(
                    f"Warning: Param mismatch ({d['mean'].shape[0]} vs {self.n_params}). Fresh start."
                )
                return False
            self.mean = d["mean"]
            self.std = d["std"]
            self.best_mean = float(d["best_mean"])
            self.gen = int(d["gen"])
            return True
        except Exception as e:
            print(f"Load failed: {e}")
            return False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = CEMTrainer(device)
    print(f"Device: {device} | Params: {trainer.n_params:,}")

    if trainer.load():
        print(f"Resumed: gen {trainer.gen}, best mean {trainer.best_mean:.1f}")
    else:
        print("Starting fresh")

    print("Warming up JIT...")
    get_moves(np.zeros((20, 10), dtype=np.float32), 0)
    print("Warmup complete.")

    while True:
        scores, improved, duration = trainer.train_gen()
        if improved:
            trainer.save("tetris_cem_best.npz")
        if trainer.gen % 25 == 0:
            trainer.save(f"tetris_cem_gen{trainer.gen:04d}.npz")
        print(
            f"Gen {trainer.gen:4d} | "
            f"Mean: {scores.mean():7.1f} | "
            f"Max: {scores.max():7.1f} | "
            f"Best mean: {trainer.best_mean:7.1f} | "
            f"{duration:5.1f}s{' *' if improved else ''}"
        )


if __name__ == "__main__":
    main()
