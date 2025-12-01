# Contribution Guidelines

Thank you for considering contributing to the Elevator Dispatch RL Project! We welcome all forms of contributions, including bug reports, feature requests, documentation improvements, and code contributions.

## Getting Started

### Prerequisites

- Python 3.7+
- Git

### Setup

1. Fork the repository
2. Clone your fork locally:

   ```bash
   git clone https://github.com/DevadattaP/elevator_dispatching_using_rl.git
   cd elevator_dispatching_using_rl
    ```

3. Install dependencies:

   You can install the exact versions used in this project by running:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, feel free to install any compatible version of the libraries if you prefer to work with a different setup.

   Note: I am using the GPU version of PyTorch. If you're on CPU-only or want a different CUDA version, please install the appropriate build from the official PyTorch instructions: <https://pytorch.org/get-started/locally/>

### Contribution Workflow

#### Reporting Issues

1. Check existing issues to avoid duplicates
2. Use the appropriate issue template:
   - 🐛 Bug Report
   - ✨ Feature Request
   - 📚 Documentation
3. Include:
   - Clear description
   - Reproduction steps (for bugs)
   - Expected vs actual behavior

#### Making Code Contributions

1. Create a new branch:

   ```bash
    git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards
3. Write/update tests
4. Update documentation if needed
5. Commit with a descriptive message:

   ```bash
    git commit -m "feat: add peak hour traffic patterns"
   ```

6. Push to your fork:

   ```bash
    git push origin feature/your-feature-name
    ```

7. Open a Pull Request against our main branch

### Areas Needing Contribution

#### High Priority

- Real-world traffic pattern datasets
- Additional RL algorithm implementations
- Performance optimization

#### Intermediate

- Enhanced visualization features
- Documentation translations
- Tutorial notebooks

#### Beginner Friendly

- Typo fixes
- Test coverage improvements
- Example configurations

## Community Guidelines

- Be respectful and inclusive
- Use welcoming language
- Assume positive intent
- Keep discussions focused on the project

### Thank you for helping make this project better! 🎉
