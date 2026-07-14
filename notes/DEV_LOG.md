# Dev Log

## Day 0
got the scaffolding done, no real implementation

## Day 1
Work on getting single GPU training working

## Day ??? Jun 17
got the dataloader working, trained for 20 steps

## Day ??? Jun 18
Wired in the attn mask
add packing and remove attention mask for now
step=14 || loss=tensor(8.3120, device='cuda:0') || tps=17924.840979258188
--> mfu so low :(
step=36 || loss=tensor(8.0759, device='cuda:0') || tps=33741.09590184431
-> changing to sdpa which uses flash attention under the hood
doubles tps
