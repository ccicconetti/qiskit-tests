"""One bit sum to demonstrate Quantum Supremacy"""

from random import randint
from time import sleep

from emoji import emojize

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    execute,
    BasicAer
)
from utils import (
    decode_message,
    NoiseModelWrapper
)

#
# Configuration
#

good_emoji = emojize(':thumbs_up:')
bad_emoji = emojize(':slightly_frowning_face:')
interactive = True
experiment_type = 'simulator-noise'
shots = 100
a = randint(0, 1)
b = randint(0, 1)

# User input (if interactive) overwrites random a and b
if interactive:
    # generated via http://patorjk.com/software/taag/
    print("\n\n\n")
    print("""
 __                 ___                __        __   __   ___             __         
/  \ |  |  /\  |\ |  |  |  |  |\/|    /__` |  | |__) |__) |__   |\/|  /\  /  ` \ /    
\__X \__/ /~~\ | \|  |  \__/  |  |    .__/ \__/ |    |  \ |___  |  | /~~\ \__,  |     
                                                                                      

""")
    print("[how to sum two bits, with style]")
    print("")
    a = int(input('Insert first  number (0 or 1): '))
    b = int(input('Insert second number (0 or 1): '))
    print("\n[contacting IBM quantum experience, loading model and circuit]\n")

#
# Execution
#

# Load noisy gates, if required
noise_wrapper = None
if experiment_type == 'simulator-noise':
    noise_wrapper = NoiseModelWrapper('ibmq_essex', quiet=True)

count_good = 0
for run in range(shots):

    # Initialize circuit and registers
    qr = QuantumRegister(2) 
    qc = QuantumCircuit(qr)

    # Prepare input
    if a == 0:
        qc.iden(0)
    else:
        qc.x(0)
    if b == 0:
        qc.iden(1)
    else:
        qc.x(1)
    qc.barrier()

    # Do the sum
    qc.cx(0, 1)

    # Measure output
    qc.measure_all()

    actual_shots = shots if not interactive else 1

    if experiment_type == 'simulator-qasm':
        backend = BasicAer.get_backend('qasm_simulator')
        result = execute(qc, backend, shots=actual_shots).result()

    elif experiment_type == 'simulator-noise':
        assert noise_wrapper is not None
        result = noise_wrapper.execute(qc, shots=actual_shots)

    # Find measurement with maximum probability
    res = decode_message(result)

    if interactive:
        sleep(1.0/(run+1))
        good = a + b == int(res[0])
        if good:
            count_good += 1
        print("attempt#{:03}: {} + {} = {} {}".format(
            run,
            a,
            b,
            res[0],
            good_emoji if good else bad_emoji
        ))

    else:
        print("{} + {} = {} (prob. {:.2f})".format(
            a,
            b,
            res[0],
            100 * float(result.get_counts()[res]) / shots))
        break

if interactive:
    sleep(1)
    print('\nWell, got it right {} out of {}, that\'s great!'.format(
        count_good,
        shots
    ))
    sleep(1)
    print('\nThis is...')
    sleep(1)
    print('\nthe *** QUANTUM SUPREMACY ***\n')