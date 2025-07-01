# Contributing to Dark Matter

Welcome to the Dark Matter project! We're excited to have you contribute to the future of RL-LLM development tools. This guide will help you get started with contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.11+
- Node.js 20+
- Git
- Docker (optional, for containerized development)

### Quick Start

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/dark-matter.git
   cd dark-matter
   ```

2. **Set Up Development Environment**
   ```bash
   # Backend setup
   cd v8.2
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
   # Frontend setup
   cd web_ui
   npm install
   ```

3. **Run the Development Servers**
   ```bash
   # Terminal 1: Backend API
   cd v8.2/backend_api
   python main.py
   
   # Terminal 2: Frontend
   cd v8.2/web_ui
   npm run dev
   ```

4. **Verify Installation**
   - Backend API: http://localhost:8000
   - Frontend UI: http://localhost:5173
   - API Documentation: http://localhost:8000/docs

## Development Setup

### Environment Configuration

Create a `.env` file in the project root:
```env
# Development settings
DEBUG=true
LOG_LEVEL=debug

# Database settings
DATABASE_URL=sqlite:///dark_matter.db

# Blockchain settings
BLOCKCHAIN_NETWORK=development
VALIDATOR_PRIVATE_KEY=your_private_key_here

# API settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:5173
```

### IDE Setup

#### VS Code
Recommended extensions:
- Python
- Pylance
- ES7+ React/Redux/React-Native snippets
- Tailwind CSS IntelliSense
- GitLens

#### PyCharm
Configure Python interpreter to use the virtual environment created above.

### Docker Development (Optional)

```bash
# Build development containers
docker-compose -f docker-compose.dev.yml build

# Start development environment
docker-compose -f docker-compose.dev.yml up

# Run tests in container
docker-compose -f docker-compose.dev.yml exec backend pytest
```

## Project Structure

```
dark-matter/
├── v8.2/
│   ├── dark_matter/           # Core Dark Matter module
│   │   ├── manager.py         # Main orchestration logic
│   │   ├── models.py          # Data models
│   │   ├── blockchain/        # Blockchain implementation
│   │   ├── green_state.py     # Canonical state management
│   │   ├── blue_state.py      # Experimental state management
│   │   └── data_contracts/    # API schemas
│   ├── backend_api/           # FastAPI backend
│   │   ├── routers/           # API route handlers
│   │   ├── cli/               # Command-line tools
│   │   └── main.py            # Application entry point
│   ├── web_ui/                # React frontend
│   │   ├── src/
│   │   │   ├── components/    # React components
│   │   │   ├── hooks/         # Custom React hooks
│   │   │   ├── pages/         # Page components
│   │   │   └── store/         # State management
│   │   └── package.json
│   ├── docs/                  # Documentation
│   ├── monitoring/            # Monitoring configuration
│   └── tests/                 # Test suites
└── README.md
```

## Development Workflow

### Branch Strategy

We use a simplified Git flow:
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical bug fixes

### Creating a Feature

1. **Create Feature Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Develop Your Feature**
   - Write code following our coding standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run backend tests
   cd v8.2
   pytest
   
   # Run frontend tests
   cd web_ui
   npm test
   
   # Run integration tests
   pytest tests/dark_matter_integration_tests/
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new environment visualization feature"
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(blockchain): add state promotion validation
fix(ui): resolve multiverse graph rendering issue
docs(api): update endpoint documentation
test(manager): add unit tests for environment creation
```

## Coding Standards

### Python Code Style

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use type hints for all function parameters and return values
- Use docstrings for all public functions and classes

```python
from typing import List, Optional, Dict, Any

def create_environment(
    base_env_id: Optional[str] = None,
    mutations: Optional[Dict[str, Any]] = None
) -> str:
    """Create a new environment with optional base and mutations.
    
    Args:
        base_env_id: ID of the base environment to clone from
        mutations: Dictionary of mutations to apply
        
    Returns:
        The ID of the newly created environment
        
    Raises:
        ValidationError: If the base environment doesn't exist
    """
    pass
```

### JavaScript/React Code Style

We use ESLint and Prettier for code formatting:
- Use functional components with hooks
- Use TypeScript for type safety (when applicable)
- Follow React best practices

```jsx
import React, { useState, useEffect } from 'react';
import { Button } from '../ui/button';

interface EnvironmentCardProps {
  environment: Environment;
  onAction: (action: string, env: Environment) => void;
}

const EnvironmentCard: React.FC<EnvironmentCardProps> = ({ 
  environment, 
  onAction 
}) => {
  const [isLoading, setIsLoading] = useState(false);

  const handleAction = async (action: string) => {
    setIsLoading(true);
    try {
      await onAction(action, environment);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="card">
      <h3>{environment.name}</h3>
      <Button onClick={() => handleAction('fork')} disabled={isLoading}>
        Fork Environment
      </Button>
    </div>
  );
};

export default EnvironmentCard;
```

### Code Quality Tools

#### Python
```bash
# Install development dependencies
pip install black isort flake8 mypy pytest

# Format code
black .
isort .

# Lint code
flake8 .
mypy .

# Run tests
pytest
```

#### JavaScript/React
```bash
# Install development dependencies
npm install --save-dev eslint prettier @typescript-eslint/parser

# Format code
npm run format

# Lint code
npm run lint

# Run tests
npm test
```

## Testing Guidelines

### Test Structure

We use a comprehensive testing strategy:
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete user workflows

### Writing Tests

#### Python Unit Tests
```python
import pytest
from dark_matter.manager import DarkMatterManager

class TestDarkMatterManager:
    def setup_method(self):
        self.manager = DarkMatterManager()
    
    def test_create_environment(self):
        env_id = self.manager.create_env()
        assert env_id is not None
        assert env_id in self.manager.environments
    
    def test_fork_environment(self):
        # Create base environment
        base_id = self.manager.create_env()
        
        # Fork it
        fork_id = self.manager.fork_env(base_id)
        
        assert fork_id != base_id
        assert fork_id in self.manager.environments
```

#### React Component Tests
```jsx
import { render, screen, fireEvent } from '@testing-library/react';
import EnvironmentCard from './EnvironmentCard';

describe('EnvironmentCard', () => {
  const mockEnvironment = {
    id: 'env_123',
    name: 'Test Environment',
    status: 'active'
  };

  const mockOnAction = jest.fn();

  test('renders environment name', () => {
    render(
      <EnvironmentCard 
        environment={mockEnvironment} 
        onAction={mockOnAction} 
      />
    );
    
    expect(screen.getByText('Test Environment')).toBeInTheDocument();
  });

  test('calls onAction when fork button is clicked', () => {
    render(
      <EnvironmentCard 
        environment={mockEnvironment} 
        onAction={mockOnAction} 
      />
    );
    
    fireEvent.click(screen.getByText('Fork Environment'));
    expect(mockOnAction).toHaveBeenCalledWith('fork', mockEnvironment);
  });
});
```

### Test Coverage

We aim for:
- **Unit Tests**: 90%+ coverage
- **Integration Tests**: Cover all major workflows
- **E2E Tests**: Cover critical user paths

```bash
# Generate coverage report
pytest --cov=dark_matter --cov-report=html
open htmlcov/index.html
```

## Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Include type hints for better IDE support
- Document complex algorithms and business logic

### API Documentation

We use FastAPI's automatic documentation generation:
- Add descriptions to all endpoints
- Include request/response examples
- Document error conditions

```python
@router.post("/create", response_model=dict)
def create_env(env_data: EnvironmentCreate):
    """Create a new environment.
    
    Creates a new environment with optional base environment and mutations.
    The environment will be created in Blue State for experimentation.
    
    Args:
        env_data: Environment creation parameters
        
    Returns:
        Dictionary containing the new environment ID and status
        
    Raises:
        HTTPException: If base environment doesn't exist (404)
        HTTPException: If validation fails (400)
    """
    pass
```

### User Documentation

- Keep documentation up to date with code changes
- Include practical examples and use cases
- Provide troubleshooting guides

## Submitting Changes

### Pull Request Process

1. **Ensure Tests Pass**
   ```bash
   # Run all tests
   pytest
   npm test
   ```

2. **Update Documentation**
   - Update relevant documentation files
   - Add docstrings for new functions
   - Update API documentation if needed

3. **Create Pull Request**
   - Use descriptive title and description
   - Reference related issues
   - Include screenshots for UI changes

4. **Code Review**
   - Address reviewer feedback
   - Ensure CI/CD checks pass
   - Maintain clean commit history

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)
Include screenshots for UI changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:
- Be respectful and constructive in discussions
- Welcome newcomers and help them get started
- Focus on what's best for the community
- Show empathy towards other community members

### Getting Help

- **Documentation**: Check the docs first
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Discord**: Join our community Discord server

### Reporting Issues

When reporting bugs:
1. Use the issue template
2. Provide minimal reproduction steps
3. Include environment details
4. Add relevant logs and screenshots

### Feature Requests

When requesting features:
1. Describe the problem you're trying to solve
2. Explain why existing solutions don't work
3. Provide detailed requirements
4. Consider implementation complexity

## Development Tips

### Performance Optimization

- Profile code before optimizing
- Use appropriate data structures
- Implement caching where beneficial
- Monitor memory usage

### Security Considerations

- Validate all inputs
- Use parameterized queries
- Implement proper authentication
- Follow security best practices

### Debugging

#### Backend Debugging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug prints
logger = logging.getLogger(__name__)
logger.debug(f"Processing environment: {env_id}")
```

#### Frontend Debugging
```jsx
// Use React Developer Tools
console.log('Component state:', state);

// Add error boundaries
class ErrorBoundary extends React.Component {
  // Error handling logic
}
```

### Common Pitfalls

- **Async/Await**: Remember to await async operations
- **State Management**: Avoid direct state mutations
- **Error Handling**: Always handle potential errors
- **Memory Leaks**: Clean up event listeners and subscriptions

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes

### Release Checklist

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Create release branch
5. Tag release
6. Deploy to staging
7. Deploy to production
8. Announce release

Thank you for contributing to Dark Matter! Your contributions help make RL-LLM development more accessible and powerful for everyone.

