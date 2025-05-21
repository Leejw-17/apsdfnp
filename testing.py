왕똥
import pygame
import tkinter as tk
from tkinter import ttk
import random, math
import matplotlib.pyplot as plt
import threading

# --- 개별 사람을 나타내는 클래스 ---
class Person:
    def __init__(self, x, y, vx, vy, radius=5):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.status = "S"  # S: 감염 가능, I: 감염, R: 회복
        self.infection_timer = 0

    def move(self, width, height):
        self.x += self.vx
        self.y += self.vy

        if self.x <= 0 or self.x >= width:
            self.vx *= -1
        if self.y <= 0 or self.y >= height:
            self.vy *= -1

    def draw(self, screen):
        colors = {"S": (0, 0, 255), "I": (255, 0, 0), "R": (0, 200, 0)}
        pygame.draw.circle(screen, colors[self.status], (int(self.x), int(self.y)), self.radius)


# --- SIR 시뮬레이션 클래스 ---
class SIRSimulation:
    def __init__(self, beta, recovery_time, population, mask_effectiveness, social_distancing, movement_restriction, width=800, height=600):
        self.base_beta = beta
        self.recovery_time = recovery_time
        self.population = population
        self.mask_effectiveness = mask_effectiveness
        self.social_distancing = social_distancing
        self.movement_restriction = movement_restriction
        self.width = width
        self.height = height
        self.people = []
        self.infection_radius = 10
        self.running = True
        self.time_step = 0
        self.history = []

    def adjust_infection_rate(self):
        beta = self.base_beta
        if self.mask_effectiveness:
            beta *= (1 - self.mask_effectiveness)
        if self.social_distancing:
            beta *= 0.7
        return beta

    def adjust_movement(self, vx, vy):
        if self.movement_restriction:
            return vx * 0.5, vy * 0.5
        return vx, vy

    def initialize(self):
        self.people = []
        for _ in range(self.population):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            vx, vy = self.adjust_movement(random.uniform(-1, 1), random.uniform(-1, 1))
            self.people.append(Person(x, y, vx, vy))
        self.people[0].status = "I"  # 초기 감염자

    def update(self, dt):
        for person in self.people:
            person.move(self.width, self.height)
            if person.status == "I":
                person.infection_timer += dt
                if person.infection_timer > self.recovery_time:
                    person.status = "R"

        beta = self.adjust_infection_rate()

        for p1 in self.people:
            if p1.status != "I":
                continue
            for p2 in self.people:
                if p2.status != "S":
                    continue
                dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
                if dist < self.infection_radius and random.random() < beta:
                    p2.status = "I"

        # 상태 기록
        counts = {"S": 0, "I": 0, "R": 0}
        for person in self.people:
            counts[person.status] += 1
        self.history.append((counts["S"], counts["I"], counts["R"]))

    def show_graph(self):
        s, i, r = zip(*self.history)
        plt.stackplot(range(len(s)), s, i, r, labels=["Susceptible", "Infected", "Recovered"], colors=["#0bb", "#f66", "#444"])
        plt.legend(loc="upper right")
        plt.xlabel("Time Steps")
        plt.ylabel("Population")
        plt.title("SIR Model Dynamics")
        plt.show()

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("SIR 모델 시뮬레이션 with 정책")
        clock = pygame.time.Clock()

        self.initialize()

        while self.running:
            dt = clock.tick(60)
            screen.fill((0, 0, 0))

            self.update(dt)

            for person in self.people:
                person.draw(screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            pygame.display.flip()

        pygame.quit()
        self.show_graph()


# --- Tkinter GUI ---
def start_simulation():
    try:
        beta = float(entry_beta.get())
        recovery_time = int(entry_recovery.get())
        population = int(entry_population.get())
        mask = mask_var.get()
        distancing = distancing_var.get()
        movement = movement_var.get()

        mask_effectiveness = 0.5 if mask else 0.0

        def run_sim():
            sim = SIRSimulation(beta, recovery_time, population, mask_effectiveness, distancing, movement)
            sim.run()

        threading.Thread(target=run_sim).start()

    except ValueError:
        print("입력값을 숫자로 입력해주세요.")

# GUI 설정
root = tk.Tk()
root.title("SIR 시뮬레이션 설정")

fields = [
    ("감염률 (β, 예: 0.2)", "0.2"),
    ("회복 시간 (ms)", "3000"),
    ("인구 수", "100")
]

entries = []
for i, (label_text, default) in enumerate(fields):
    ttk.Label(root, text=label_text).grid(row=i, column=0, sticky="e", padx=5, pady=5)
    entry = ttk.Entry(root)
    entry.insert(0, default)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries.append(entry)

entry_beta, entry_recovery, entry_population = entries

mask_var = tk.BooleanVar()
ttk.Checkbutton(root, text="마스크 착용", variable=mask_var).grid(row=3, column=0, sticky="w")

distancing_var = tk.BooleanVar()
ttk.Checkbutton(root, text="사회적 거리두기", variable=distancing_var).grid(row=4, column=0, sticky="w")

movement_var = tk.BooleanVar()
ttk.Checkbutton(root, text="인구 이동 제한", variable=movement_var).grid(row=5, column=0, sticky="w")

# 실행 버튼
ttk.Button(root, text="시뮬레이션 시작", command=start_simulation).grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()
