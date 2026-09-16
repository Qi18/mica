# M00b r3正式前置门禁

Dense/MoE四卡DDP连续12步与真实SIGTERM6步+独立resume至12步比较：全部模型/optimizer tensor差为0（361/681项）。
41个正式数据分片3,804,930,498字节SHA均一致。
两臂full-data100步profile尚在执行，未将此报告标记completed；完整validation与正式启动状态待实际证据更新。
源代码与配置指纹见source-manifest.json，使用确定性计算、固定DDP桶与NCCL Ring/Simple；原阈值未放宽。

## 门禁最终结果

两臂4卡恢复最大差均0；full-data100步均完成，完整validation均11525行/6400000 targets，两个checkpoint strict load通过。
Dense约0.669s/update、峰值12959MiB；MoE约0.942s/update、峰值16723MiB。
此处batch32x4/accum2，与M00 batch4的50.5%吞吐比不是同一个配置。正式质量结论仍待训练与七项Base评测。
状态：completed（仅前置门禁完成，不是Phase7收尾）。
