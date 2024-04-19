import csv

def writeCSV(filename, x, y, xlabel, ylabel):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([xlabel, ylabel])
        for i in range(len(x)):
            writer.writerow([x[i], y[i]])

def exportCSV(Domain_L, Domain_S):
    # Reading Positions in Domain_L and Domain_S
    pos_L = Domain_L.position
    pos_S = Domain_S.position + Domain_L.L
    # Reading Velocities in Domain_L and Domain_S
    vel_L = Domain_L.v
    vel_S = Domain_S.v
    
    writeCSV('Square_Dvorak_v_L.csv', pos_L, vel_L, 'pos_L', 'vel_L')
    writeCSV('Square_Dvorak_v_S.csv', pos_S, vel_S, 'pos_S', 'vel_S')



