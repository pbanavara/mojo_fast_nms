### Mojo Fast NMS
Public repo for the AGIHouse Modular hackathon project

#### Objective
Implement a faster NMS kernel on par with Cutlass::FastNMS. This is to help speed up inference on YOLO models.

#### Algorithm
Key idea (Barrios et al., 2025): load a tile of 32 candidate boxes into shared memory, build a suppression mask per box, then use warp-wide __ballot_sync() + bit-tricks to knock out overlaps. 
This amortizes memory traffic and avoids atomics because each warp works on independent tiles.
[Paper](https://arxiv.org/html/2502.00535v1?utm_source=chatgpt.com)

#### Status
Does not compile - still fixing compilation errors

#### Complexity
Still O(n^2) but GPU advantages in computing IoUs in parallel. used a maske


