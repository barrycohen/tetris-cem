import numpy as np
import random
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

SHAPES = {
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
PIECES = list(SHAPES.keys())
COLORS = {
    "I": (0, 240, 240),
    "O": (240, 240, 0),
    "T": (160, 0, 240),
    "S": (0, 240, 0),
    "Z": (240, 0, 0),
    "J": (0, 0, 240),
    "L": (240, 160, 0),
}
SCORE = [0, 40, 100, 300, 1200]


class BoardEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 2))
        self.fc1 = nn.Linear(64 * 4 * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

    def set_params_flat(self, flat_params):
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data = torch.FloatTensor(
                flat_params[idx : idx + size].reshape(p.shape)
            ).to(p.device)
            idx += size


class Bag:
    def __init__(self):
        self.q = []

    def next(self):
        if not self.q:
            self.q = PIECES[:]
            random.shuffle(self.q)
        return self.q.pop()


def get_drops(board, piece):
    drops = []
    for shape in SHAPES[piece]:
        min_c = min(c for _, c in shape)
        max_c = max(c for _, c in shape)
        for x in range(-min_c, 10 - max_c):
            y = 0
            while True:
                stop = False
                for r, c in shape:
                    ny = y + r + 1
                    if ny >= 20 or (ny >= 0 and board[ny, x + c]):
                        stop = True
                        break
                if stop:
                    break
                y += 1
            if any(y + r < 0 for r, c in shape):
                continue
            # FIX: Check if the final position overlaps with existing pieces
            if any(board[y + r, x + c] for r, c in shape if y + r >= 0):
                continue
            b = board.copy()
            for r, c in shape:
                b[y + r, x + c] = 1
            full = np.all(b, axis=1)
            cleared = int(np.sum(full))
            if cleared:
                b = np.vstack([np.zeros((cleared, 10), dtype=np.float32), b[~full]])
            drops.append((b, cleared, [(y + r, x + c) for r, c in shape]))
    return drops


def greedy_select(net, device, board, cur):
    drops = get_drops(board, cur)
    if not drops:
        return None
    with torch.no_grad():
        boards = np.array([d[0] for d in drops])
        t = torch.FloatTensor(boards).unsqueeze(1).to(device)
        scores = net(t).cpu().numpy()
    for i, (_, cl, _) in enumerate(drops):
        scores[i] += cl * 10
    return drops[np.argmax(scores)]


def lookahead_select(net, device, board, cur, nxt):
    drops = get_drops(board, cur)
    if not drops:
        return None
    all_b, idx, rew = [], [], []
    for i, (b1, cl1, _) in enumerate(drops):
        d2 = get_drops(b1, nxt)
        if not d2:
            all_b.append(b1)
            idx.append(i)
            rew.append(cl1 * 10 - 500)
        else:
            for b2, cl2, _ in d2:
                all_b.append(b2)
                idx.append(i)
                rew.append((cl1 + cl2) * 10)
    if not all_b:
        return None
    with torch.no_grad():
        t = torch.FloatTensor(np.array(all_b)).unsqueeze(1).to(device)
        sc = net(t).cpu().numpy() + np.array(rew)
    idx = np.array(idx)
    best = [sc[idx == i].max() if (idx == i).any() else -1e9 for i in range(len(drops))]
    return drops[np.argmax(best)]


def main():
    d = np.load("tetris_cem_best.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BoardEvaluator().to(device)
    net.eval()
    net.set_params_flat(d["mean"])
    print(f"Gen {int(d['gen'])}, avg {float(d['best_mean']):.1f} lines")

    pygame.init()
    C = 30
    screen = pygame.display.set_mode((C * 10 + 220, C * 20 + 40))
    pygame.display.set_caption("Tetris AI")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24, bold=True)
    fontB = pygame.font.SysFont("Arial", 36, bold=True)
    fontS = pygame.font.SysFont("Arial", 16)

    grid = [[None] * 10 for _ in range(20)]
    bag = Bag()
    use_bag = False
    use_lookahead = True
    cur = bag.next()
    nxt = bag.next()
    lines = score = 0
    speed, paused, over, last = 100, False, False, 0

    def board():
        return np.array(
            [[1.0 if c else 0.0 for c in row] for row in grid], dtype=np.float32
        )

    def next_piece():
        return bag.next() if use_bag else random.choice(PIECES)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    grid = [[None] * 10 for _ in range(20)]
                    bag = Bag()
                    cur, nxt = next_piece(), next_piece()
                    lines = score = 0
                    over = False
                elif e.key == pygame.K_b:
                    use_bag = not use_bag
                elif e.key == pygame.K_l:
                    use_lookahead = not use_lookahead
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    speed = max(0, speed - 50)
                elif e.key == pygame.K_MINUS:
                    speed = min(500, speed + 50)

        now = pygame.time.get_ticks()
        if not over and not paused and (speed == 0 or now - last >= speed):
            b = board()
            if use_lookahead:
                res = lookahead_select(net, device, b, cur, nxt)
            else:
                res = greedy_select(net, device, b, cur)
            if res is None:
                over = True
            else:
                _, cl, cells = res
                for r, c in cells:
                    grid[r][c] = cur
                if cl:
                    rows = [i for i in range(20) if all(grid[i])]
                    for r in sorted(rows, reverse=True):
                        del grid[r]
                        grid.insert(0, [None] * 10)
                    lines += cl
                    score += SCORE[cl]
                cur, nxt = nxt, next_piece()
                if not get_drops(board(), cur):
                    over = True
            last = now

        screen.fill((20, 20, 20))

        for y in range(20):
            for x in range(10):
                c = grid[y][x]
                rect = pygame.Rect(20 + x * C, 20 + y * C, C - 1, C - 1)
                pygame.draw.rect(screen, COLORS[c] if c else (40, 40, 40), rect)
        pygame.draw.rect(screen, (80, 80, 80), (18, 18, C * 10 + 4, C * 20 + 4), 2)

        sx = C * 10 + 40

        screen.blit(font.render("NEXT", True, (255, 255, 255)), (sx, 20))
        pygame.draw.rect(screen, (40, 40, 40), (sx, 50, 100, 60))
        pygame.draw.rect(screen, (80, 80, 80), (sx, 50, 100, 60), 2)
        for r, c in SHAPES[nxt][0]:
            pygame.draw.rect(
                screen, COLORS[nxt], (sx + 10 + c * 20, 55 + r * 20, 18, 18)
            )

        screen.blit(font.render("SCORE", True, (255, 255, 255)), (sx, 130))
        screen.blit(fontB.render(str(score), True, (255, 255, 100)), (sx, 158))

        screen.blit(font.render("LINES", True, (255, 255, 255)), (sx, 220))
        screen.blit(fontB.render(str(lines), True, (100, 255, 100)), (sx, 248))

        col = (100, 255, 100) if use_bag else (255, 100, 100)
        screen.blit(
            font.render(f"7-Bag {'ON' if use_bag else 'OFF'}", True, col), (sx, 310)
        )

        col = (100, 255, 100) if use_lookahead else (255, 100, 100)
        screen.blit(
            font.render(f"Lookahead {'ON' if use_lookahead else 'OFF'}", True, col),
            (sx, 340),
        )

        screen.blit(
            font.render(
                f"Speed {'MAX' if speed == 0 else f'{speed}ms'}", True, (180, 180, 180)
            ),
            (sx, 375),
        )

        if paused:
            screen.blit(font.render("PAUSED", True, (255, 255, 0)), (sx, 415))

        cy = 500
        for key, label in [
            ("+/-", "Speed"),
            ("B", "Toggle 7-Bag"),
            ("L", "Toggle Lookahead"),
            ("Space", "Pause"),
            ("R", "Restart"),
            ("Esc", "Quit"),
        ]:
            screen.blit(fontS.render(key, True, (180, 180, 180)), (sx, cy))
            screen.blit(fontS.render(label, True, (120, 120, 120)), (sx + 50, cy))
            cy += 20

        if over:
            ov = pygame.Surface((C * 10, C * 20))
            ov.set_alpha(180)
            ov.fill((0, 0, 0))
            screen.blit(ov, (20, 20))
            t = fontB.render("GAME OVER", True, (255, 50, 50))
            screen.blit(t, (20 + C * 5 - t.get_width() // 2, C * 8))
            t = font.render(f"{lines} lines", True, (255, 255, 255))
            screen.blit(t, (20 + C * 5 - t.get_width() // 2, C * 10))
            t = font.render(f"Score: {score}", True, (255, 255, 255))
            screen.blit(t, (20 + C * 5 - t.get_width() // 2, C * 11 + 10))
            t = fontS.render("R to restart", True, (180, 180, 180))
            screen.blit(t, (20 + C * 5 - t.get_width() // 2, C * 13))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()

