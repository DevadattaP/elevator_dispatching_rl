# Elevator Dispatch System with Reinforcement Learning

## Abstract

This project explores the application of Reinforcement Learning (RL) to optimize elevator dispatching in multi-floor, multi-elevator buildings. Traditional rule-based elevator control systems often struggle to adapt to dynamic traffic patterns and optimize multiple objectives simultaneously. I developed a comprehensive simulation environment modeling realistic elevator physics, and various traffic patterns. I implemented multiple RL algorithms including PPO, A2C, DQN, SAC, TD3, DDPG, and their variants and evaluated them against traditional rule-based systems. The system incorporates enhanced state representations, diverse reward functions, and specialized action spaces inspired by recent research. Performance was assessed using metrics such as average waiting time, passenger throughput, elevator utilization, and fairness. While RL agents achieved comparable performance to rule-based systems, challenges in energy efficiency were identified, suggesting the need for hybrid approaches combining learning with domain constraints.

## Objectives

- Create a detailed elevator simulation with realistic physics, door operations, and passenger behavior patterns.
- Train and compare various RL agents including value-based, policy-based, and actor-critic methods.
- Develop enhanced observation spaces incorporating elevator states, passenger information, and traffic patterns.
- Design and test multiple reward structures balancing efficiency, fairness, and energy considerations.
- Establish comprehensive evaluation metrics including passenger waiting times, throughput, and system efficiency.
- Benchmark RL performance against conventional rule-based elevator control systems.

## Motivation

Reinforcement learning (RL) offers a principled framework for learning control policies through interaction with the environment, making it a strong candidate for problems involving uncertainty, sequential decisions, and long-term rewards. Historically, RL has been successfully applied to several influential benchmark problems, including Samuel’s checkers player, TD-Gammon, the Acrobot, dynamic channel allocation, and job-shop scheduling---with \emph{elevator dispatching} explicitly recognized as one of the canonical industrial case studies.

Elevator control is described as:
> ''A good example of a stochastic optimal control problem of economic importance that is too large to solve by classical techniques such as dynamic programming. Waiting for an elevator is a situation familiar to all of us. How long we wait depends on the dispatching strategy: if passengers on several floors request pickups, which should be served first, and how should elevators position themselves when idle?''[[5]](http://incompleteideas.net/book/ebook/node111.html)

Elevator dispatching represents a complex sequential decision-making problem under uncertainty. Traditional control systems rely heavily on handcrafted heuristic rules that, while efficient in predictable conditions, lack the adaptability required to handle varying traffic patterns, changing passenger behavior, and multi-objective optimization needs. The elevator group control problem (EGCP) is characterized by:

- **High-dimensional state space:** Multiple elevators, diverse floor configurations, and stochastic passenger arrivals
- **Dynamic environment:** Strong temporal variation in traffic intensity (up-peak, down-peak, lunchtime, mixed)
- **Multiple competing objectives:** Minimizing waiting time, journey time, and energy consumption while maintaining fairness
- **Real-time decision making:** Policies must react within milliseconds to new hall calls and car events

Despite its presence in RL literature for decades, large-scale deployment of RL-based elevator control has been limited. Existing work often focuses on simplified environments, restricted action spaces, or handcrafted reward functions, leaving several open questions:

- Why are modern RL methods (PPO, SAC, TD3, DQN variants) not yet widely adopted in real elevator systems?
- Do contemporary algorithms actually outperform well-tuned rule-based dispatchers under realistic conditions?
- What practical limitations---energy usage, fairness, stability, training complexity---prevent real-world deployment?

These gaps formed the motivation for this work. After reviewing the literature, I became particularly interested in understanding whether modern RL algorithms could meaningfully improve elevator dispatching and what challenges would arise in practice. Moreover, despite the long-standing interest in elevator control as an RL benchmark, I found no publicly available, complete, or actively maintained implementations of modern RL approaches for this problem. This lack of accessible code further motivated the development of a fully reproducible simulation and training framework, systematic evaluation of multiple RL families, and analysis of their strengths, limitations, and deployment feasibility as part of this work.

> [!NOTE]
> Visit [report](./report/report.pdf) for detailed discussion regarding the work donw, results, conclusion and future scope.

## References

[1] N. Vaartjes and V. Francois-Lavet, “Novel rl approach for efficient elevator group control systems,” arXiv preprint arXiv:2507.00011, p. 15, 2025, <https://doi.org/10.48550/arXiv.2507.00011>.

[2] J. Wan, K. Lee, and H. Shin, “Traffic pattern-aware elevator dispatching via deep reinforcement learning,” Advanced Engineering Informatics, vol. 61, p. 102497, 2024, <https://doi.org/10.1016/j.aei.2024.102497>.

[3] J. Sorsa, H. Ehtamo, M.-L. Siikonen, T. Tyni, and J. Ylinen, “The elevator dispatching problem,” Transportation Science, 2009, <https://www.researchgate.net/profile/Marja-Liisa-Siikonen/publication/228964635> The Elevator Dispatching Problem.

[4] X. Yuan, L. Bus¸oniu, and R. Babuˇska, “Reinforcement learning for elevator control,” IFAC Proceedings Volumes, vol. 41, no. 2, pp. 2212–2217, 2008, <https://doi.org/10.3182/20080706-5-KR-1001.00373>.

[5] M. Lee, “Elevator dispatching case study,” 2005, <http://incompleteideas.net/book/ebook/node111.html>.

[6] R. H. Crites and A. G. Barto, “Improving elevator performance using reinforcement learning,” Advances in Neural Information Processing Systems, vol. 8, pp. 1017–1023, 1995, <https://proceedings.neurips.cc/paper_files/paper/1995/file/390e982518a50e280d8e2b535462ec1f-Paper.pdf>

## How to Contribute

We welcome contributions through:

- Pull requests (with accompanying tests)
- Issue reports (bug/feature)
- Real-world traffic pattern datasets
- Algorithm benchmarking results

See our [Contribution Guidelines](CONTRIBUTING.md) for details.

> [!Note]
> The future scope mentioned reflects our current development priorities. Specific features may change based on community feedback and research advancements.
