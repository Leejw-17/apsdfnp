import pygame
import tkinter as tk
from tkinter import ttk
import random, math
import matplotlib.pyplot as plt
import threading
import numpy as np
import os
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
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
        #인구 모형의 현실적 움직임을 위해 방향 전환 알고리즘 추가함
    def draw(self, screen):
        colors = {"S": (0, 0, 255), "I": (255, 0, 0), "R": (0, 200, 0)}
        pygame.draw.circle(screen, colors[self.status], (int(self.x), int(self.y)), self.radius)
        # s:감염가능자, i:감염자, r:회복자 를 각각 blue, red, green으로 나타냄
#SIR 시뮬레이션 클래스 
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
        self.infection_radius = 20
        self.running = True
        self.time_step = 0
        self.history = []
        self.r0_history = []

    def adjust_infection_rate(self):
        beta = self.base_beta
        if self.mask_effectiveness:
            beta *= (1 - self.mask_effectiveness)
        if self.social_distancing:
            beta *= (1 - 0.4)
            # 질병관리청의 통계에 근거하여 사회적 거리두기 정책을 시행한후 감염재생산지수가 2배 가량 감소했다는 정보를 이용하여 감염률을 50%감소시킴
        return beta

    def adjust_movement(self, vx, vy):
        if self.movement_restriction:
            vx *= (1 - 0.25)
            vy *= (1 - 0.25)
        return vx, vy
        # 통계청의 자료에 근거하여 인구 이동 제한 정책을 시행한 후 교통사고 비율이 정책 전 대비 26% 감소했다는 결과를 이용해,
        # 교통사고 비율은 인구 이동에 밀접한 연관이 있기에 정책 시행시 인구 모형의 이동속도를 25% 감소시켰다.

    def initialize(self):
        # 랜덤으로 인구 모형의 움직임을 지정함
        self.people = []
        for _ in range(self.population):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            vx, vy = self.adjust_movement(random.uniform(-1, 1), random.uniform(-1, 1))
            self.people.append(Person(x, y, vx, vy))
        # 랜덤으로 초기 감염자를 지정함
        self.people[0].status = "I"  

    def update(self, dt):
        for person in self.people:
            person.move(self.width, self.height)
            if person.status == "I":
                person.infection_timer += dt
                if person.infection_timer > self.recovery_time:
                    person.status = "R"
        #시간 측정해서 그래프에 반영함
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

        # sir의 현재 상태를 기록하는 부분
        counts = {"S": 0, "I": 0, "R": 0}
        for person in self.people:
            counts[person.status] += 1
        self.history.append((counts["S"], counts["I"], counts["R"]))

        # R0 추정치를 계산하는 코드 : (신규 감염자 / 현재 감염자)
        current_I = counts["I"]
        r0 = (new_infections / current_I) if current_I > 0 else 0
        self.r0_history.append(r0*10)

    #그래프 출력 함수
    def show_graph(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        s, i, r = zip(*self.history)
        time = range(len(s))

        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        #왼쪽: SIR 비율을 면적으로 나타낸 그래프를 출력하는 코드
        axes[0].stackplot(time, s, i, r,
                         labels=['Susceptible', 'Infected', 'Recovered'],
                         colors=['#1f77b4', '#ff7f0e', '#2c2c2c'])  
        # 그래프 내에서 파랑, 주황, 회색은 각각 감염가능자, 감염자, 회복자의 수를 나타냄
        axes[0].set_title("SIR Model Dynamics", fontsize=14)
        axes[0].set_xlabel("Time Steps")
        axes[0].set_ylabel("Population")
        axes[0].legend(loc='lower right')
        axes[0].grid(False)
        axes[0].annotate('Removed',
                         xy=(1.01, 0.5), xycoords='axes fraction',
                         va='center', ha='left',
                         fontsize=12, rotation=90,
                         color='white',
                         annotation_clip=False)

        #오른쪽: R₀ 이동 평균 그래프를 출력하는 코드
        window_size = 10
        r0_avg = moving_average(self.r0_history, window_size)
        avg_time = range(window_size - 1, len(self.r0_history))

        axes[1].plot(avg_time, r0_avg, color='orange', linewidth=2, label=f'Moving Avg R₀ (w={window_size})')
        axes[1].axhline(y=1, color='gray', linestyle='--', linewidth=1)

        # 평균 R₀ 값 계산하는 코드
        mean_r0 = np.mean(self.r0_history)
        # 그래프 우측 상단의 평균 R₀ 값을 텍스트로 표시함
        axes[1].text(0.95, 0.9, f"평균 R₀: {mean_r0:.2f}", 
             transform=axes[1].transAxes, 
             ha='right', va='top', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

        axes[1].set_title("Estimated R₀ Over Time", fontsize=14)
        axes[1].set_xlabel("Time Steps")
        axes[1].set_ylabel("R₀")
        axes[1].legend(loc='upper right')
        axes[1].grid(False)

        plt.tight_layout()
        plt.show()


        # SIR 곡선 출력하는 코드
        plt.subplot(1, 2, 1)
        plt.plot(time, s, label='Susceptible', color='blue')
        plt.plot(time, i, label='Infected', color='red')
        plt.plot(time, r, label='Recovered', color='green')
        plt.title('SIR Model Dynamics')
        plt.xlabel('Time Steps')
        plt.ylabel('Population')
        plt.legend()

        # R₀ 이동 평균 그래프를 출력하는 부분
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

def start_simulation():
    try:
        beta = float(entry_beta.get())
        recovery_time = int(entry_recovery.get())
        population = int(entry_population.get())
        mask = mask_var.get()
        distancing = distancing_var.get()
        movement = movement_var.get()
        mask_effectiveness = 0.6 if mask else 0.0
        #학술자료에 근거하면 천마스크의 감염예방효과는 56%, 수술용 마스크는 66%, N95 또는 KF94 마스크는 83%의 감염병 예방효과가 있다는 정보를 얻었다.
        #이를 근거로 마스크의 감염예방효과를 60%로 설정하였다.
        def run_sim():
            global sim_history, sim_r0
            sim = SIRSimulation(beta, recovery_time, population, mask_effectiveness, distancing, movement)
            sim.run()
            sim_history = sim.history
            sim_r0 = sim.r0_history

        threading.Thread(target=run_sim).start()

    except ValueError:
        print("입력값을 숫자로 입력해주세요.")
       
# R₀값의 평균을 계산하는 과정
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

root = tk.Tk()
root.title("SIR 시뮬레이션 설정")

fields = [
    ("감염률 (β, 예: 0.02)", "0.02"),
    ("회복 시간 (ms)", "3000"),
    ("인구 수", "3000")
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
ttk.Button(root, text="시뮬레이션 시작", command=start_simulation).grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()