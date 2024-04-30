import csv

def writeCSV(filename, x, y, xlabel, ylabel):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([xlabel, ylabel])
        for i in range(len(x)):
            writer.writerow([x[i], y[i]])

def exportCSV(csv_L, csv_S, Domain_L, Domain_S):
    # Reading Positions in Domain_L and Domain_S
    pos_L = Domain_L.position
    pos_S = Domain_S.position + Domain_L.L
    # Reading Velocities in Domain_L and Domain_S
    vel_L = Domain_L.v
    vel_S = Domain_S.v
    
    writeCSV(csv_L, pos_L, vel_L, 'pos_L', 'vel_L')
    writeCSV(csv_S, pos_S, vel_S, 'pos_S', 'vel_S')


def vHalftoCSV(t_L, v_L, v_s, t_prev, v_L_prev, v_s_prev, pos_L, pos_s, length_L):

    if (t_L > 0.00150 and t_L < 0.001502):
        # Find Vel values at Full Time Step with mean
        prev_vel_L = v_L_prev
        prev_vel_S = v_s_prev
        prev_t_L = t_prev
        vel_L = (prev_vel_L + v_L) / 2
        vel_S = (prev_vel_S + v_s) / 2
        writeCSV('Square_New_v_LTEST_wBulk.csv', pos_L, vel_L, 'pos_L', 'vel_L' )
        writeCSV('Square_New_v_STEST_wBulk.csv', pos_s + length_L, vel_S, 'pos_S', 'vel_S')
        print("CSV Time: ", (t_L + prev_t_L) / 2)
