# Contributing to The Annotation Garden Project

Thank you for your interest in contributing to this VLM-based image annotation system for Natural Scene Dataset (NSD) research!

## Getting Started

### Development Setup

1. Fork and clone the repository
2. Set up the development environment:
```bash
conda activate torch-312  # or create: conda create -n torch-312 python=3.12
pip install -e .
```

3. Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

### Project Structure

- `src/image_annotation/` - Core Python package
- `frontend/` - Next.js web dashboard
- `tests/` - Test suite (real data only, NO MOCKS)
- `.context/` - Development context files
- `.rules/` - Development standards and patterns

## Development Workflow

### Branching Strategy

1. Create feature branches: `git checkout -b feature/short-description`
2. Make atomic commits with descriptive messages (<50 chars, no emojis)
3. Test thoroughly before pushing
4. Submit pull requests to main branch

### Code Standards

#### Python
- Follow `.rules/python.md` standards
- Use ruff for formatting: `ruff check --fix . && ruff format .`
- Type hints required for all functions
- Real tests only - absolutely NO MOCKS (see `.rules/testing.md`)

#### Testing Philosophy
- Use real NSD images for testing
- Real OLLAMA/LLM API calls only
- Docker containers for test databases
- Test against actual behavior, not mocked interfaces

#### Documentation
- Examples over explanations
- Keep README concise - details go in separate docs
- Update `.context/` files for development context

## Contribution Types

### Code Contributions
- Bug fixes and feature implementations
- Performance optimizations
- Test coverage improvements
- Documentation updates

### Research Contributions
- New VLM model integrations
- Annotation quality improvements
- NSD processing enhancements
- Performance benchmarking

### Infrastructure
- CI/CD improvements
- Docker configurations
- Deployment optimizations

## Pull Request Process

1. **Check Context**: Review `.context/plan.md` for current priorities
2. **Research**: Update `.context/research.md` if exploring new approaches
3. **Document Failures**: Log attempts in `.context/scratch_history.md`
4. **Test**: Run `pytest tests/ --cov` with real data
5. **Format**: Run `ruff check --fix . && ruff format .`
6. **Commit**: Atomic commits, descriptive messages
7. **PR**: Reference relevant issues and context files

### PR Requirements
- All tests pass with real data
- Code coverage maintained or improved
- Documentation updated where needed
- No breaking changes without discussion
- Performance implications considered

## Issue Guidelines

### Bug Reports
- Include environment details (Python version, OS, dependencies)
- Provide minimal reproduction case with real data
- Include relevant log output
- Reference specific NSD images if applicable

### Feature Requests
- Describe the research use case
- Consider impact on 25k+ annotation processing
- Discuss integration with existing VLM models
- Provide implementation suggestions if possible

## Development Best Practices

### Performance Considerations
- System handles 25k+ annotations efficiently
- Database queries optimized for large datasets
- Memory usage monitored during batch processing
- Token usage tracking for cost management

### Security
- No API keys in code or commits
- Environment variables for sensitive configuration
- Secure handling of research data

### Research Ethics
- Respect NSD dataset usage guidelines
- Consider annotation quality and bias
- Document model performance characteristics

## Getting Help

- Check `.context/` files for current development context
- Review `.rules/` directory for detailed standards
- Open issues for questions or discussions
- Reference existing code patterns in the codebase

## Recognition

Contributors will be acknowledged in:
- README.md contributor section
- Release notes for significant contributions
- Research publications where applicable (with permission)

---

By contributing, you agree that your contributions will be licensed under the project's CC-BY-NC-SA 4.0 license.