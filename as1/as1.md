Question 1.1
  - Combined degree is calculated with both in and out degree of a node.
  - Using notion of reverse neigborhood: N ′(v) = {u ∈V : (u, v) ∈E}
  - deg(v) = deg^- (v) + deg^+ (v)
  - where deg^- (v) is in-degree and and latter (deg^+(v)) is outdegree. Indegree being set of nodes N'(v) where (u,v) ∈ E and in outdegree (v,u) ∈ E
Question 1.2
  - Nk(W ) = N (Nk−1(W )) ∪Nk−1(W ) is redudant since every node is connected to every other node.
  - No need for iteration of k>1 since every node is reachable within k=1
  - We can say the the for each node the neigborhood will contain every other node. N(v) = V. To be precise we have to substract the node itself so N(v) = V - {v}
  Question 1.3
  - Since k-neighborhood represent set of nodes with k-reach of a node (let say node {v}) and reverse neighborhood is set of nodes connected to node {v}. Combining them would give us a set of nodes that are k-reachable and connected to node {v}.
  - It measures how does the node {v} influence other nodes within k-reach.
  Question 1.4
  - r(G, v) = {u ∈ N'(v) : (v, u) ∈ E} / N'(v)
  - r (G,V) is reciprocity of directed network G on specific node.How many of connections "come back" to {v}
  - {u ∈ N'(v) : (v, u) ∈ E}: set of all nodes (u) that are from reverse neighorhood (direct incoming connection/degree)

  Question 1.5
  - SCC (Strongly Connected Components) algorhitm - Kosaraju based
  1. Initialize nodes and set status to "not visited"
  2. DFS traversal of graph starting from node {v} if DFS doesn't visit all nodes containg {u} then return false.
  3. Reset status of all visited nodes to "not  visited" 
  4. Repeat step 2. but with starting node {u}. IF DFS doesn't visit all nodes containg {v} then return false
  5. If both DFS from {u} and {v} visit same nodes then return true (both nodes are from SCC).

  https://www.geeksforgeeks.org/connectivity-in-a-directed-graph/

  Question 1.6
  1. For each node {v} in network:
  - get the neighborhood (using k-neighborhood for k=1) - nodes that are connected to {v}
  2. For each pair {u,w} in neighborhood check if there's edge. If there is edge then numberOfTrinagles += 1.
  3. Repeat for all nodes.
  4. Divide numberOfTriangles by 3 as each trinagle is counted 3 times.
  
  Time  complexity: 
  Iteration for all nodes in network = 0(V).
  Interation for each node to find out neigborhood = O(V).
  Pairs in neighborhood can in be worst case = O(V^2).
  Therefore total time complexity can be O(V^4) in worst case. Although apparently in practise it's more like O(V^3) and can be reduced to O(V^2.8074) using Strassen's matrix multiplication.

  https://www.geeksforgeeks.org/number-of-triangles-in-a-undirected-graph/

  Question 1.7
  
  1. For each node {v} in network:
  - Count degree of node
  2. For each neigbor {u} of node {v}:
  - Count degree of node
  - if degree{u} > degree{v} then count += 1
  3. Repeat for each node in network and each neighbor for said node.
  4. paradox_false = count - number_of_nodes

  paradox_false variable will hold number of counts where Friendship paradox does not hold true.

  Time complexity:
  Iteration for all nodes in network = 0(V).
  Iteration for each neigbor is equivalent to amount of edges therefore 0(E)
  Step 3. is just O(1)
  Total complexity therefore is O(E*V). Depending on amount of edges in network.

Question 1.8

 1. Deleting an edge {v,w} ∈ E
 - both nodes will loose one edge therefore (n'-1, m'-1)
 - neigbors of either {v} or {w} will simply loose connection to the nodes therefore (n', m'-1)
 2. Adding an edge {v,w} ∈/E
 - Both nodes will gain one edge therefore (n'+1, m'+1)
 - neigbors of either {v} or {w} will simply gain connection to the nodes therefore (n', m'+1)

 Unless the node is connected to affected node said node will not feel a change of state.

 Question 1.9

Radius R(G) = minu∈V e(u) = minimal eccentricity
Basically calculating the longest direct route with the fewest edgest from one node to another.

1. Calculate eccentrity of each node in V. (either BFS or Dijkstr)
2. The biggest value of shortest path (maximum shortest distance) is the value of radius (minimal eccentricity).
