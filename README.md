# CLAP-SMS: Cross Lingual Audio Pre-training championing Speech, Music and Sound
In the realm of multi-modal learning, audio plays a crucial role, but the scarcity of
detailed annotations hinders effective training. Contrastive learning has shown promise
for text-to-audio retrieval and audio classification tasks, yet several key challenges
remain unaddressed. Audio’s continuous and variable-length nature leads to temporal
misalignment when paired with discrete text tokens, resulting in inefficient or lossy
fixed-window processing. Furthermore, the cross-modal semantic gap between audio
spectrograms and text tokens complicates similarity learning, often exacerbated by
fixed temperature scaling. Additionally, reliance solely on contrastive loss can cause
modality collapse, where audio and text embeddings lose their unique modality traits.
Attention mechanisms commonly face sparsity and misalignment, while domain biases
arise from random sampling across heterogeneous datasets. To address these, we
propose , a comprehensive framework featuring: (1) an adaptive variable-length
processing pipeline leveraging an Adaptive CNN with Dynamic Pooling and intelligent
audio pre-processing including silence trimming and repetition handling; (2) learnable
similarity calibration via adaptive sigmoid parameters to optimize cross-modal
alignment; (3) anti-collapse modality classification using dual classifiers to maintain
modality-specific information while aligning embeddings; (4) bidirectional cross-modal
attention with explicit alignment loss for improved feature relevance; and (5) a multi-
domain balanced sampling strategy with domain prefixing to mitigate domain bias and
boost generalization. Rigorous experiments on text-to-audio retrieval and audio
classification—both zero-shot and supervised—demonstrate that CLAP-SMS achieves
state-of-the-art performance, rivalling fully supervised models and advancing robust
audio representation learning.
