# Contributing to AW-SafeSeg

Thank you for your interest in contributing to the AW-SafeSeg project!

## Project Structure

Please familiarize yourself with the project structure:

```
pro/
├── project/          # Main application (backend + frontend)
├── IDDAW/           # Training dataset
├── full4/           # Alternative training setup
├── docs/            # Documentation
└── scripts/         # Utility scripts
```

## Development Setup

1. **Fork and clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   cd project/frontend && npm install
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and small (<50 lines)

### JavaScript/React Code
- Use ES6+ syntax
- Follow Airbnb style guide
- Use functional components with hooks
- Add JSDoc comments for complex functions

## Testing

Before submitting a pull request:

1. **Test backend**:
   ```bash
   python scripts/test_integration.py
   ```

2. **Test frontend**:
   ```bash
   cd project/frontend
   npm run test
   ```

3. **Manual testing**:
   - Start the full stack application
   - Test all major features
   - Check for console errors

## Commit Messages

Use clear, descriptive commit messages:

```
feat: Add confidence threshold slider to UI
fix: Resolve memory leak in image processing
docs: Update deployment guide with Docker instructions
refactor: Simplify model loading logic
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description

## Areas for Contribution

- **Model improvements**: Better architectures, optimization
- **Frontend features**: UI/UX enhancements
- **Documentation**: Tutorials, examples, translations
- **Testing**: Unit tests, integration tests
- **Performance**: Optimization, caching, parallelization
- **Deployment**: Docker, Kubernetes, cloud platforms

## Questions?

- Check existing documentation in `docs/`
- Review closed issues and PRs
- Open a new issue for discussion

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

Thank you for contributing! 🙏
