# Neurobiomorphic AI System

Advanced biologically-inspired AI system combining neural plasticity, reinforcement learning, language reasoning, and hierarchical task structures. This system achieves human-like performance on complex multi-task learning benchmarks.

**ALWAYS follow these instructions first and only fallback to additional search and context gathering if the information here is incomplete or found to be in error.**

## Working Effectively

### Bootstrap and Environment Setup
- Create virtual environment: `python3 -m venv venv`
- Activate virtual environment: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
- Install dependencies: `pip install -r requirements.txt`
  - **NEVER CANCEL**: Installation takes ~2-15 minutes depending on network speed. Set timeout to 20+ minutes.
  - Installs PyTorch, Transformers, Gymnasium, NumPy, Matplotlib and CUDA dependencies
  - May encounter network timeouts - retry if installation fails
- Validate installation: `python test_components.py`
  - Should complete in ~5 seconds and show "✓ All tests passed!"

### Build and Test
- **No traditional build step required** - Python modules are imported directly
- Run component tests: `python test_components.py` (5 seconds)
- Run minimal system demo: `python minimal_demo.py` 
  - **NEVER CANCEL**: Takes ~10-15 seconds. Set timeout to 30+ seconds.
  - Shows full system integration without requiring model downloads
- Run full experiment: `python experiments/run_experiments.py`
  - **CRITICAL**: Requires internet access to download GPT-2 model (~500MB) on first run
  - **NEVER CANCEL**: Model download takes 5-10 minutes, full training takes 30+ minutes
  - Set timeout to 60+ minutes for full experiments

### Validation Scenarios
- **Always run `python test_components.py`** after making changes to core components
- **Always run `python minimal_demo.py`** to validate system integration
- **Test full language reasoning**: Run experiments with network access to validate GPT-2 integration
- **Manual validation**: Check that all 3 episodes complete successfully in minimal demo

### Code Quality
- **Always run linting**: `flake8 src/ experiments/ --max-line-length=100 --ignore=E203,W503`
- **Always run formatting**: `black src/ experiments/ --line-length=100`
- Fix linting issues before committing - CI will fail on style violations

## Key Projects and Components

### Core Architecture (`src/`)
1. **Neural Plasticity** (`src/neural_plasticity/`)
   - `neuroplastic_network.py`: Main neuroplastic network with multiple plasticity layers
   - `advanced_plasticity_layer.py`: Advanced plasticity with astrocyte modulation, STDP, dendritic computation
   - Implements: astrocyte modulation, spike-timing-dependent plasticity, synaptic tagging, homeostatic plasticity

2. **Reinforcement Learning** (`src/reinforcement_learning/`)
   - `advanced_rl_agent.py`: Actor-critic RL agent with experience replay
   - `actor_critic.py`: Neural network architecture for actor-critic
   - Uses PyTorch, memory replay, gradient clipping

3. **Language Reasoning** (`src/language_reasoning/`)
   - `language_reasoner.py`: GPT-2 based reasoning and feature extraction
   - **CRITICAL**: Downloads GPT-2-medium model (~500MB) on first use
   - Generates explanations and extracts 768-dimensional features

4. **Hierarchical Agent** (`src/hierarchical_agent/`)
   - `hierarchical_agent.py`: Two-level hierarchical architecture
   - `low_level_policy.py`: Low-level action policy conditioned on goals
   - `high_level_policy.py`: High-level goal generation policy
   - Handles different cognitive process levels

5. **Hybrid Learning** (`src/hybrid_learning/`)
   - `enhanced_hybrid_system.py`: Integrates all components into unified system
   - Combines neuroplastic networks, RL, language reasoning, and hierarchical structure

### Experiments (`experiments/`)
- `run_experiments.py`: Main experiment script using BipedalWalker-v3 environment
- Runs 1000 episodes of training with periodic model updates
- **NEVER CANCEL**: Full experiment takes 30+ minutes

### Test and Demo Files
- `test_components.py`: Validates core components without model downloads (5 seconds)
- `minimal_demo.py`: End-to-end system demo with mock environment (10-15 seconds)

## Common Tasks and Patterns

### Working with Neural Plasticity
- Networks expect combined input: `torch.cat([state, language_features], dim=-1)`
- Context tensors should be 768-dimensional (GPT-2 hidden size)
- Plasticity layers modify weights during forward passes (learning while running)

### Working with Language Reasoning
- **First run requires internet** to download GPT-2 model
- Use `MockLanguageReasoner` for offline testing (see `minimal_demo.py`)
- Always handle feature extraction: `language_features = reasoner.extract_features(text)`

### Working with Hierarchical Agents
- Goal dimensions must match between high-level and low-level policies
- State dimensions: typically 24 (environment state size)
- Action dimensions: typically 4 (BipedalWalker actions)
- Goal dimensions: typically 8 (configurable)

### Integration Testing
- Always test component integration with `minimal_demo.py` after changes
- Check tensor dimension compatibility between components
- Validate that episodes complete successfully without crashes

## Important Code Patterns

### Tensor Dimensions
- State vectors: `[batch_size, state_dim]` typically `[1, 24]`
- Language features: `[batch_size, 768]` (GPT-2 hidden size)
- Combined input: `[batch_size, state_dim + 768]` typically `[1, 792]`
- Actions: `[batch_size, action_dim]` typically `[1, 4]`
- Goals: `[batch_size, goal_dim]` typically `[1, 8]`

### Common Issues and Solutions
- **Dimension mismatch errors**: Check tensor shapes match expected dimensions above
- **Import errors**: Always add `sys.path.append()` for relative imports in scripts
- **Model download failures**: Ensure internet access or use mock components for testing
- **Memory issues**: Reduce batch sizes if OOM errors occur

### File Organization
- All core implementations in `src/[component]/` directories
- Experiments and demos in root directory
- Always maintain `__init__.py` files for proper module imports

## Time Expectations
- **NEVER CANCEL** any of these operations:
- Virtual environment creation: 30 seconds
- Dependency installation: 2-15 minutes (includes PyTorch + CUDA, varies with network)
- Component tests: 5 seconds
- Minimal demo: 10-15 seconds  
- GPT-2 model download: 5-10 minutes (first run only, requires internet)
- Full experiment: 30-60 minutes (1000 episodes)

### Network Requirements
- **Dependency installation**: Requires internet access to PyPI
- **Model downloads**: GPT-2 requires internet access on first run (~500MB download)
- **Offline testing**: Use `test_components.py` and `minimal_demo.py` which work without internet

## Debugging and Troubleshooting

### Component Testing
- Import errors: Check `__init__.py` files and relative imports
- Tensor shape errors: Validate dimensions match expected patterns above
- CUDA errors: System will fallback to CPU automatically

### Performance Issues  
- Slow training: Normal - neural plasticity involves real-time weight updates
- Memory issues: Reduce batch sizes in RL agent and hybrid system
- Network timeouts: 
  - PyPI timeouts during installation: Retry `pip install -r requirements.txt`
  - GPT-2 download failures: Retry or use mock language reasoner for testing
  - Use `--timeout 1000` flag for pip if needed: `pip install --timeout 1000 -r requirements.txt`

### Integration Issues
- Component communication: Check tensor dimension compatibility
- Episode failures: Validate environment reset and step functions
- Training instability: Normal for complex multi-component system

Always validate changes with both `test_components.py` and `minimal_demo.py` before considering work complete.