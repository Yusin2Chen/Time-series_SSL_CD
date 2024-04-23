# Time-series_SSL_CD
***a novel approach to unsupervised change detection in satellite image time-series*** leveraging the power of contrastive learning and feature tracking. While existing methods have primarily focused on bi-temporal change detection, our approach expands this scope to address the challenges posed by image time-series with seasonal changes.

Traditionally, state-of-the-art models for image time-series change detection have relied on clustering learning or training models from scratch with tailored pseudo labels, limiting their ability to generalize across diverse scenarios. In contrast, our two-stage methodology combines contrastive learning with feature tracking to exploit spatial-temporal information effectively.

In the first stage, we derive pseudo labels from pre-trained models and utilize feature tracking to propagate them within the image time-series, enhancing label consistency and resilience to seasonal variations. We then employ a self-training algorithm with ConvLSTM, integrating supervised contrastive loss and contrastive random walks to refine feature correspondence in space-time.

Finally, a fully connected layer is fine-tuned on the pre-trained multi-temporal features to generate precise change maps. Our approach is validated through extensive experiments on two datasets, demonstrating consistent enhancements in accuracy for both fitting and inference scenarios.

By releasing our code on GitHub, we aim to facilitate further research and development in the field of unsupervised change detection in satellite imagery. Our method not only pushes the boundaries of existing techniques but also provides a robust framework for addressing real-world challenges in remote sensing applications. Join us in advancing the capabilities of satellite image analysis and contribute to the evolution of this exciting field.

