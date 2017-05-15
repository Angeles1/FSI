# Search methods

import search

ab = search.GPSProblem('O', 'N', search.romania)

print "---------------------Busqueda en Anchura----------"

print search.breadth_first_graph_search(ab).path()
print "---------------------Busqueda en Profundidad----------"

print search.depth_first_graph_search(ab).path()
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()

print "---------------------Branch and Bound----------"
print search.ramificacion_y_acotacion(ab).path()

print "---------------------Branch and Bound subestimated--"
print search.ramificacion_y_acotacion_subestimacion(ab).path()




#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450
