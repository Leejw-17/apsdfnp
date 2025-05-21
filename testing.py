
import pygame
import tkinter as tk
from tkinter import ttk
import random, math
import matplotlib.pyplot as plt
import threading
import numpy as np
import os

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
        self.infection_radius = 20  # 히트박스 크기 확장
        self.running = True
        self.time_step = 0
        self.history = []
        self.r0_history = []

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

        new_infections = 0
        for p1 in self.people:
            if p1.status != "I":
                continue
            for p2 in self.people:
                if p2.status != "S":
                    continue
                dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
                if dist < self.infection_radius and random.random() < beta:
                    p2.status = "I"
                    new_infections += 1

        # 상태 기록
        counts = {"S": 0, "I": 0, "R": 0}
        for person in self.people:
            counts[person.status] += 1
        self.history.append((counts["S"], counts["I"], counts["R"]))

        # R0 추정치 추가 (신규 감염자 / 현재 감염자)
        current_I = counts["I"]
        r0 = (new_infections / current_I) if current_I > 0 else 0
        self.r0_history.append(r0)

    def save_simulation(self, filename="sir_simulation_data.npz"):
        data = {
            "history": np.array(self.history),
            "r0_history": np.array(self.r0_history),
        }
        np.savez_compressed(filename, **data)
        print(f"시뮬레이션이 '{filename}'에 저장되었습니다.")

    def load_simulation(self, filename="sir_simulation_data.npz"):
        if not os.path.exists(filename):
            print(f"파일 '{filename}'을 찾을 수 없습니다.")
            return False
        data = np.load(filename)
        self.history = data["history"].tolist()
        self.r0_history = data["r0_history"].tolist()
        print(f"시뮬레이션 데이터를 '{filename}'에서 불러왔습니다.")
        return True

    def show_graph(self):
        s, i, r = zip(*self.history)
        time = range(len(s))

        plt.figure(figsize=(12, 5))

        # SIR 곡선
        plt.subplot(1, 2, 1)
        plt.plot(time, s, label='Susceptible', color='blue')
        plt.plot(time, i, label='Infected', color='red')
        plt.plot(time, r, label='Recovered', color='green')
        plt.title('SIR Model Dynamics')
        plt.xlabel('Time Steps')
        plt.ylabel('Population')
        plt.legend()

        # R0 이동 평균 그래프
        plt.subplot(1, 2, 2)
        window_size = 10
        r0_avg = moving_average(self.r0_history, window_size)
        avg_time = range(window_size - 1, len(self.r0_history))
        plt.plot(avg_time, r0_avg, label=f'Moving Avg R₀ (w={window_size})', color='orange')
        plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
        plt.title('Estimated R₀ Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('R₀')
        plt.legend()

        plt.tight_layout()
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

         # 실행 후 결과 저장 (전역 변수 사용)
        global sim_history, sim_r0
        sim_history = sim.history
        sim_r0 = sim.r0_history

        threading.Thread(target=run_sim).start()

    except ValueError:
        print("입력값을 숫자로 입력해주세요.")
# R0 평균 계산
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

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
# --- 시뮬레이션 저장 및 불러오기 기능 ---

# 저장된 시뮬레이션 기록을 위한 전역 변수
sim_history = []
sim_r0 = []

# 시뮬레이션 저장
def save_simulation():
    sim = SIRSimulation(0.2, 3000, 0, 0, False, False)  # dummy 초기화
    sim.history = sim_history
    sim.r0_history = sim_r0
    sim.save_simulation()

# 시뮬레이션 불러오기
def load_simulation():
    sim = SIRSimulation(0.2, 3000, 0, 0, False, False)
    if sim.load_simulation():
        sim.show_graph()

# 저장 및 불러오기 버튼 추가 (시작 버튼 아래)
ttk.Button(root, text="시뮬레이션 저장", command=save_simulation).grid(row=7, column=0, padx=5, pady=5)
ttk.Button(root, text="시뮬레이션 불러오기", command=load_simulation).grid(row=7, column=1, padx=5, pady=5)

root.mainloop()

