import random
import numpy as np
import matplotlib.pyplot as plt

# Environment size
width = 5
height = 16

# Actions
num_actions = 4

actions_list = {"UP": 0,
                "RIGHT": 1,
                "DOWN": 2,
                "LEFT": 3
                }
#bordes
actions_vectors = {"UP": (-1, 0),
                   "RIGHT": (0, 1),
                   "DOWN": (1, 0),
                   "LEFT": (0, -1)
                   }
# Discount factor
discount = 0.8

#creacion de la tabla Q llena de ceros, con el altoXancho y numero de acciones
Q = np.zeros((height * width, num_actions))  # Q matrix
Rewards = np.zeros(height * width)  # Reward matrix, it is stored in one dimension

#Funciones de utilidad para la tabla
#transforma a Snumestado
def getState(y, x):
    return y * width + x


def getStateCoord(state):
    return int(state / width), int(state % width)

#dado un estado dice las acciones que se pueden hacer dentro de ese estado, para los bordes
def getActions(state):
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append("RIGHT")
    if x > 0:
        actions.append("LEFT")
    if y < height - 1:
        actions.append("DOWN")
    if y > 0:
        actions.append("UP")
    return actions

#dado un estado coge las acciones que se pueden hacer en ese estado y selecciona aleatoriamente 1 para poder explorar
def getRndAction(state):
    return random.choice(getActions(state))


def getRndState():
    return random.randint(0, height * width - 1)

#se rellena la tabla con las recompensas
Rewards[4 * width + 3] = -10000
Rewards[4 * width + 2] = -10000
Rewards[4 * width + 1] = -10000
Rewards[4 * width + 0] = -10000

Rewards[9 * width + 4] = -10000
Rewards[9 * width + 3] = -10000
Rewards[9 * width + 2] = -10000
Rewards[9 * width + 1] = -10000
#celda final objetivo la 3,3 con recompensa 100
Rewards[3 * width + 3] = 100
final_state = getState(3, 3)

print np.reshape(Rewards, (height, width))


def qlearning(s1, a, s2):
    Q[s1][a] = Rewards[s2] + discount * max(Q[s2])
    return

#Siempre se busca el de mayor recompensa y se explora ese camino,llevando a la explotacion
def getGreedyAction(state):
    if max(Q[state]) > 0:
        a = np.argmax(Q[state])
        if a == 0:
            return "UP"
        if a == 1:
            return "RIGHT"
        if a == 2:
            return "DOWN"
        if a == 3:
            return "LEFT"
    else:
        return getRndAction(state)


#opcion e-greedy con un 10% de probabilidades dar opcion al azar. Tendras un balanceo entre exploracion(10%) y explotacion (90%)
def getEGreedyAction(state,rate):
    if(random.random()>rate):
        return getRndAction(state)
    else:
        return getGreedyAction(state)

#poner aqui variable numero_acciones totales realizadas- se incrementa para saber cual es el numero total de acciones, eje x
#si hago 100 acciones cuanta recompensa total me he llevado hasta ahora??
#quien ganara?
#politica que cuando llegue a 1000 acciones tendra 900 puntos de recompensa
#otro llega en 1000 con 300 puntos de recompensa
#hay que optimizar la politica. la peor politica de todas es la totalmente aleatoria
#nosotros tenemos que encontrar la mejor politica de todas :( posiblemente e-greedy y probar con 10%, 20%, 30% ...etc
# Episodes


recompensa_total = 0
numero_acciones = 0
for i in xrange(100):
#primero un estado aleatorio para empezar
    state = getRndState()
    #es estado final o no? EL WHILE ES AL AZAR EN LOS 100 EPISODIOS, NO SABEMOS CUANDO ACABARA HASTA QUE LLEGUE AL OBJETIVO, SE DEDICA A GENERAR ACCION ALEATORIA SIN BASARSE EN INFORMACION QUE TENGA PARA REALIZAR UNA ACCION DETERMINADA
    #TENEMOS QUE PROMEDIAR EL NUMERO DE ACCIONES QUE HACEMOS EN UN EPISODIO PARA LLEGAR AL ESTADO INICIAL. CUAL ES ESE NUMERO? podemos empezar con un estado no aleatorio, por ejemplo 0,0 (en funcion del estado que empieza el numero de acciones sera diferente) generar episodios explorando para conocer el promedio de acciones que necesitamos para alcanzar el objetivo.
    while state != final_state:
        numero_acciones+=1
    ##selecciono una accion aleatoria
	#PRIMERA MODIFCIACION EN LA LINEA DE ACTION, si no tiene informacion en la tabla q elegimos al azar
	#si tiene informacion usamos el metodo avaricioso.
	#OJO: las recompensas estan en la tabla R, informacion en la tabla Q
	#es aniadir tres o cuatro linas de codigo
	#si distinto de 0 elegir el max y coger la accion, sino random.
        action = getEGreedyAction(state,0.95)
        #action = getRndAction(state) #linea clave para el punto 1. Esta linea coge todo el rato exploracion, nunca se utiliza info de la tabla q
    #implementamos el caso de estar en un determinado estado y todas las acciones tienen valor 0, lo unico que hacer es tomar una al azar, pero
	#en el momento en que sea distinto de 0 hay informacion para llegar a algun sitio y llegar al final
	#ventajas: llegaremos en un numero determinado de acciones que conocemos.
	#desventaja: ese camino puede no ser el mejor de todos si siempre explotas ese camino
	#hay que darle opciones al azar para descubrir nuevos caminos
        y = getStateCoord(state)[0] + actions_vectors[action][0]
        x = getStateCoord(state)[1] + actions_vectors[action][1]
        #nuevo estado
        new_state = getState(y, x)
        #ACTUALIZAR EL ESTADO,  ACCTION_LIST CONVIERTE LA STRING A UN NUMERO
        qlearning(state, actions_list[action], new_state)
        state = new_state
    recompensa_total += Rewards[state]



print Q


# Q matrix plot
#IMPRIMIR, PLOTEAR EL TABLERO, NO HAY QUE TOCAR NADA, NO QUISO EXPLICARLO PORQUE NO SE NECESITA MODIFICAR

s = 0
ax = plt.axes()
ax.axis([-1, width + 1, -1, height + 1])

for j in xrange(height):

    plt.plot([0, width], [j, j], 'b')
    for i in xrange(width):
        plt.plot([i, i], [0, height], 'b')

        direction = np.argmax(Q[s])
        if s != final_state:
            if direction == 0:
                ax.arrow(i + 0.5, 0.75 + j, 0, -0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 1:
                ax.arrow(0.25 + i, j + 0.5, 0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 2:
                ax.arrow(i + 0.5, 0.25 + j, 0, 0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 3:
                ax.arrow(0.75 + i, j + 0.5, -0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
        s += 1

    plt.plot([i+1, i+1], [0, height], 'b')
    plt.plot([0, width], [j+1, j+1], 'b')

print "Promedio de acciones por capitulo: ", numero_acciones/100
print "Recompensa total en el capitulo", recompensa_total


plt.show()



#paso 1: introducir una politica para que el agente alterne entre exploracion y explotacion
#paso 2: opcion e-greedy con un 10% de probabilidades dar opcion al azar. Tendras un balanceo entre exploracion(10%) y explotacion (90%)
#paso 3: ver en un tiempo limitado si hemos maximizado la recompensa obtenida, como sabemos si maximizamos? tiempo, recompensa en eje de coordenadas
#el tiempo se puede medir en el numero de acciones que hagamos, saltar de un estado a otro consume una unidad de tiempo. 
#la recompensa se mide por el tiempo.

#NOTA: dice que son muy pocas lineas de codigo
