# Elevator Dispatch System with Reinforcement Learning

## Abstract

This project explores the use of Reinforcement Learning (RL) to address the elevator dispatching problem—determining how multiple elevators should move to efficiently serve passengers within a dynamic, multi-floor building. Traditional rule-based systems often fail to adapt to varying traffic patterns or optimize for multiple objectives at once. The goal here is to develop an adaptive, learning-based dispatch system that can intelligently balance efficiency, comfort, and energy use. The project involves building a detailed simulation environment that models realistic elevator physics, door operations, and passenger motion in various traffics like peak and non-peak hours.
The RL agent will be trained and evaluated across various configurations of building size, number of elevators, and time-based demand patterns. Some important experiments to be conducted include comparing discrete and continuous action spaces, designing and testing alternative observation space structures and reward functions, and benchmarking multiple RL algorithms such as PPO and others to identify the most effective control strategy.
System performance will be evaluated using metrics such as average waiting time of passengers, travel time distribution, elevator utilization, and estimated energy consumption. A visual interface will provide real-time insights into the agent’s decisions and overall building performance. Through these explorations, the project hopes to advance the state-of- knowledge on the use of how reinforcement learning can be applied to the challenging problems of scheduling and control problems, to create more intelligent, data-driven control systems.

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

> **Note**: This roadmap reflects our current development priorities. Specific features may change based on community feedback and research advancements.
