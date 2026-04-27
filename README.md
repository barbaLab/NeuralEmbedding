# NeuralEmbedding

**NeuralEmbedding** is a MATLAB library designed to provide a set of tools for analyzing neural dynamics. This library includes various dimensionality reduction techniques tailored for spiking neural data, along with useful metrics to evaluate the quality of the generated embeddings as well as plotting methods to visualize data.

## Features

- **Dimensionality Reduction Techniques**: Apply various algorithms to reduce the dimensionality of spiking neural data, helping to uncover underlying neural dynamics.
- **Evaluation Metrics**: Utilize built-in metrics to assess the effectiveness and quality of the generated embeddings.
- **Manifold Diagnostics**: Validate reconstructed latent spaces with shuffle-based dimension selection, cross-validated reconstruction, permutation-tested decoding, and Procrustes alignment across sessions.
- **Flexible and Extensible**: The library is designed to be flexible, allowing for easy integration with existing workflows and extending with new methods.

## Installation

To use NeuralEmbedding, clone the repository and add it to your MATLAB path:

```matlab
repositoryURL("https://github.com/yourusername/NeuralEmbedding.git",folder)
addpath(fullfile(folder,"NeuralEmbedding"));
```

## Usage

Here’s a simple example of how to use the library to perform dimensionality reduction on spiking neural data.
For details on input data formats, the available dimensionality reduction methods, and evaluation metrics, please refer to the wiki.

```matlab
% Load your neural data
load('neural_data.mat');

NE = NeuralEmbedding(data,...
  "fs",fs,...
  "time",T,...
  "area",A,...
  "condition",C,);

% Apply a dimensionality reduction technique
NE.findEmbedding("PCA");

% Evaluates trajectory length
NE.computeMetrics("arc");
```

## Documentation

For detailed documentation and examples, please refer to the [Wiki](https://github.com/barbaLab/NeuralEmbedding/wiki).

### Manifold Diagnostics

The library includes a **manifold diagnostic toolkit** for validating latent-space quality.
See [`docs/manifold_diagnostics.md`](docs/manifold_diagnostics.md) for the full guide.

Quick-start (single session):

```matlab
% Dimension selection via parallel analysis
resA = NE.selectDimension(1:15);
fprintf('Selected dim: %d\n', resA.dStar);

% Cross-validated reconstruction
resB = NE.crossValReconstruct(resA.dStar);

% Cross-validated decoding + permutation test
resC = NE.crossValDecode(y_trial, resA.dStar);
fprintf('Acc=%.1f%%  p=%.4f\n', resC.accMean*100, resC.pValue);

% Multi-session alignment (accepts NeuralEmbedding object array)
resD = alignSessions([NE1, NE2, NE3]);
```

## Contributing

Contributions are welcome! If you’d like to contribute, please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
