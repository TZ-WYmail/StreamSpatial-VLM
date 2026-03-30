南方科技大学本科生毕业设计（论文）任务书
设计（论文）题目	StreamSpatial-VLM：基于2D-3D互引导的流式空间理解高效框架
学生姓名	谭政	学号	12212320	专业	计算机科学与技术	院系	计算机科学与工程系
主要任务及基本要求（包括设计或研究的内容、要求与指标、应完成的成果、进程安排及主要参考文献目录等）：

一、设计与研究内容
本课题旨在研发一个名为 StreamSpatial-VLM 的高效流式空间理解框架。核心研究内容是通过2D-3D双向引导机制，解决现有Spatial-VLM在计算效率与流式处理上的瓶颈。具体设计与研究内容包括：
系统瓶颈分析与基线构建：复现并剖析当前主流Spatial-VLM（如VG-LLM），精确定量其在帧级和token级的计算冗余，为后续优化提供数据驱动的基线。
2D→3D语义门控机制设计：研究并设计一个轻量级门控模块。该模块需能依据当前视频帧与历史帧的2D语义特征差异，智能决策是否激活计算密集的3D几何网络，以避免无效的跨帧计算。
3D→2D几何引导压缩算法研究：探索利用3D几何先验（如深度图方差、位姿不确定性）生成无监督重要性图的方法，并基于此实现对2D视觉编码器输出的patch token进行动态、无参裁剪。
综合实验验证与消融分析：在权威空间问答数据集上，对所提框架进行全面、系统的实验验证，包括与基线模型的性能对比、各核心模块的消融研究，以及关键参数的敏感性分析。

二、要求与指标
功能要求：
完成StreamSpatial-VLM框架的完整代码实现，支持逐帧视频流输入。框架包含2D→3D门控、3D→2D压缩和增量缓存三大核心模块，同时提供清晰的API接口，便于在SPAR-7M、ScanQA等标准数据集上进行评估。
性能要求：
在SPAR-7M数据集上，与VG-LLM等强基线相比，实现端到端推理速度提升。在实现上述加速的同时，在ScanRefer、ScanQA等零样本空间理解任务上的核心指标（如Accuracy、EM）下降不超过2%。模型在流式处理过程中的峰值显存占用应保持在合理范围（例如，低于基线模型的50%）。
技术要求：
2D→3D门控与3D→2D压缩模块不引入额外的可训练推理参数。整个框架的优化过程不依赖任何3D真值标注数据，实现自监督或弱监督优化。

三、应完成的成果
一套完整的开源项目：在GitHub等平台创建一个公开代码仓库，包含：
StreamSpatial-VLM框架的完整源代码及详细注释。项目依赖的安装说明与使用教程。预训练模型文件，便于其他研究者复现结果。所有实验的配置文件与详细结果。
一份详细的毕业设计论文：符合学校规范，全面阐述研究背景、技术方案、实验过程与结论，字数不少于规定要求。
一套答辩材料：制作内容清晰、逻辑严谨的答辩PPT，并准备一个可现场演示的流式空间理解Demo系统。

四、进程安排
前期： 完成文献调研与开题报告。复现并剖析主流基线模型，精确定位其计算瓶颈，为后续优化提供明确目标。
中期： 集中研发StreamSpatial-VLM的核心技术。完成2D→3D门控、3D→2D压缩和增量缓存三大模块的设计与实现，并搭建出可运行的流式推理系统原型。
后期： 进行全面的实验验证与分析，撰写并投稿学术论文。同时，完成毕业设计论文的撰写、代码开源以及最终的答辩准备工作。

五、主要参考文献目录
[1] VG-LLM: Visual Grounded Large Language Model for 3D Scene Understanding. *NeurIPS 2025*.
[2] Spatial-MLMM: Spatial Multi-Modal Large Model for Efficient 3D Reasoning. *arXiv preprint*.
[3] VidCom²: Budget-Constrained Adaptive Token Compression for Video Understanding. *CVPR 2024*.
[4] METok: Event-Aware Multi-Stage Token Compression for Online Video Understanding. *ICLR 2025*.
[5] DYTO: Dynamic Token Merging via Zero-Shot Hierarchical Clustering. *NeurIPS 2024*.
[6] ReKV: Retrieval-Augmented KV Cache for Streaming Video Question Answering. *ACL 2024*.
[7] ∞-Video: Understanding Videos with Continuous-Time Memory Integration. *ICML 2024*.
[8] VGGT: A Generative Video-to-Geometry Transformer for 3D Scene Reconstruction. *CVPR 2024*.
[9] SPAR-7M: A Large-Scale Spatial Reasoning Dataset for Embodied AI. *arXiv preprint*.
[10] ScanQA: Answering Questions about 3D Scenes via Scene-Language Grounding. *CVPR 2022*.
 



南方科技大学本科生毕业设计（论文）开题报告
设计（论文）题目	StreamSpatial-VLM：基于2D-3D互引导的流式空间理解高效框架
学生姓名	谭政	学号	12212320	专业	计算机科学与技术
题目类型	A.理论研究	题目来源	A.指导教师出题	指导教师	郑锋	职称	副教授（研究员）
（国内外研究概况，研究目的和意义、研究方法、思路与预期成果；任务完成的阶段内容及时间安排；完成毕业设计（论文）所具备的条件因素等）

一、	研究目的和意义
空间智能是VLM走向具身应用的关键。当前Spatial-VLM（如VG-LLM）采用“2D语义+3D几何”双分支，提升了3D场景理解精度，但代价是计算量翻倍。问题主要在三点：帧级冗余（3D网络对每帧重复计算）、token级冗余（>60%的2D patch-token与空间任务无关）、流式盲区（现有方法假设视频已缓存，无法逐帧处理）。
本课题的目标是构建一个高效流式空间理解框架。核心思路是用2D-3D互引导来动态分配计算资源。使用2D语义差异决定何时启动3D网络，再用3D几何信息（深度、位姿）指导2D token的裁剪。整个框架追求零额外推理参数、零3D标注、零整段缓存，最终在SPAR-7M等数据集上实现3倍加速，精度不降。
 
二、国内外研究概况
在3D几何增强方面，VG-LLM（NeurIPS 2025）将Qwen2.5-VL与VGGT-1B结合，在ScanRefer任务上把指标提升了9.4%，但该方法资源需求量高，且推理复杂度随帧数T线性增长。
帧采样方向，Spatial-MLMM提出贪心覆盖策略，把输入帧数减少约50%，然而每帧仍然要向ViT输入完整的patch序列，单帧token量保持O(HW)不变，计算瓶颈并未真正缓解。
Token压缩领域， 2024-2025 的压缩研究不再局限于“整段后剪枝”，而是向“在线、预算驱动”演进：VidCom² 通过两阶段自适应预算，按帧间相似度给帧配额，再在帧内做Top-K token剪枝，在不降低精度的前提下视觉 token 降到 25 %  。METok 提出事件感知多段管道：编码段进行事件切分，预填充段依据注意力强度裁剪 token，解码段持续维护压缩后的 KV-Cache，实现显存节省与帧间一致性，浮点运算减少 65 %。DYTO 零样本层次聚类合并 patch，支持动态预算，在 30 % token 下零样本 VQA 掉点 <2 % 。
流式/在线视频理解方向，近一年已出现多条与“逐帧增量推理”直接相关的内容：ReKV 提出“滑窗注意力 + 增量 KV-Cache 检索”，无需完整视频可实时回答 Streaming VQA，首次把推理复杂度从O(T)压到O(w)。OVBench & VideoChat-Online 推出“金字塔记忆库”：帧按时间-空间双层粒度下写与驱逐，与 KV-Cache 同步删除过期 token，实现真正意义上的在线视频对话。∞-Video 用连续时间记忆积分，在单遍视频流中动态调整注意粒度，长视频-QA 速度提升 2-3 倍且无额外训练。
 
尽管上述研究已验证“流式输入＋常数级记忆”的可行性，但其 token 压缩与帧选择仍局限于 RGB 外观或注意力分数，未引入深度、位姿等 3D 几何先验，难以在空间推理任务中精准分配计算预算。如何以 3D 几何信号直接引导在线压缩与帧更新，成为当前缺失的关键环节，也正是 StreamSpatial-VLM 的切入点。
 
二、	研究方法与思路
学生将承担“StreamSpatial-VLM：基于2D-3D互引导的流式空间理解高效框架”的完整研发，具体包括以下环节：
1. 复现基线：跑通 VG-LLM 与 Spatial-MLLM，统计帧-与 token-级冗余，输出 profiling 报告。  
2. 2D→3D Gate：用 2D 语义差异控制 3D 网络触发。  
3. 3D→2D Zip：基于 VGGT 深度方差与位姿不确定性生成重要性图，无参裁剪 ViT patch。  
4. 增量缓存：维护 3D 隐状态队列与 2D 特征缓存，实现逐帧流式推理。  
5. 实验与开源：在 SPAR-7M、ScanQA 零样本 benchmark 完成对比，达成 3× 加速与精度持平，撰写论文并开源代码与模型。
 
四、条件因素分析
1.	数据：SPAR-7M（700万QA对）已开源，含单视/多视/视频3种格式，无需额外采集。
2.	代码：VG-LLM与OpenCLIP已开源，接口成熟；VGGT-1B提供现成3D几何提取API，可直接调用。
3.	算力：训练：仅微调Gate与压缩策略，峰值显存<7 GB，RTX 3060-12G笔记本完成；推理：单卡，无需TensorRT/量化。
4.	评估：零样本空间benchmark（SPAR-7M、VSI-Bench）提供标准协议，无需人工标注。
 
五、工作量与阶段安排  
1.前期：文献综述、开题报告、基线复现与冗余分析、伪深度&VGGT离线生成  
2.中期：2D→3D Gate实现与调优、3D→2D Zip无参压缩+消融、增量缓存流式集成  
3.后期：SPAR-7M/ScanQA实验与可视化、论文撰写+开源、答辩PPT+实时演示
 
六、预期成果
一套完整的开源项目：在GitHub等平台创建一个公开代码仓库，包含：
StreamSpatial-VLM框架的完整源代码及详细注释。项目依赖的安装说明与使用教程。预训练模型文件，便于其他研究者复现结果。所有实验的配置文件与详细结果。
一份详细的毕业设计论文：符合学校规范，全面阐述研究背景、技术方案、实验过程与结论，字数不少于规定要求。
一套答辩材料：制作内容清晰、逻辑严谨的答辩PPT，并准备一个可现场演示的流式空间理解Demo系统。
 
 