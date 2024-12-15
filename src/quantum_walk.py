from qiskit import QuantumCircuit, Aer, execute
import numpy as np

class QuantumWalk:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.num_position_qubits = 2
        self.num_qubits = 2 * self.num_position_qubits + 2
        self.num_classical = 2 * self.num_position_qubits
        self.start_pos = (0, 0)  # Starting at origin
        self.is_complete = False
        self.completion_step = None

    def apply_position_shift(self, qc, start_qubit, control_qubit, direction=1):
        """Apply position shift operation."""
        # Apply controlled rotation to create superposition of moving and not moving
        qc.h(control_qubit)
        
        if direction == 1:
            # Increment with superposition
            qc.cx(control_qubit, start_qubit)
            qc.cx(start_qubit, start_qubit + 1)
            # Add phase for interference
            qc.p(np.pi/4, start_qubit)
        else:
            # Decrement with superposition
            qc.x(control_qubit)
            qc.cx(control_qubit, start_qubit)
            qc.cx(start_qubit, start_qubit + 1)
            # Add phase for interference
            qc.p(-np.pi/4, start_qubit)
            qc.x(control_qubit)

    def create_circuit(self, steps):
        """Create a quantum circuit for 2D quantum walk."""
        qc = QuantumCircuit(self.num_qubits, self.num_classical)
        
        # Initialize at (0,0) - no need for X gates as qubits start in |0⟩ state
        
        # Initialize coins in equal superposition
        qc.h([self.num_qubits - 2, self.num_qubits - 1])
        
        qc.barrier()
        
        for step in range(steps):
            # Alternate between x and y movements to reduce interference
            if step % 2 == 0:
                # X coordinate movement
                self.apply_position_shift(qc, 0, self.num_qubits - 2, 1)
                self.apply_position_shift(qc, 0, self.num_qubits - 2, -1)
                # Phase shift for x-direction
                qc.p(np.pi/3, [0, 1])
            else:
                # Y coordinate movement
                self.apply_position_shift(qc, 2, self.num_qubits - 1, 1)
                self.apply_position_shift(qc, 2, self.num_qubits - 1, -1)
                # Phase shift for y-direction
                qc.p(np.pi/3, [2, 3])
            
            # Coin operation with phase
            qc.h([self.num_qubits - 2, self.num_qubits - 1])
            qc.p(np.pi/2, [self.num_qubits - 2, self.num_qubits - 1])
            
            qc.barrier()
        
        # Measure position qubits
        for i in range(2 * self.num_position_qubits):
            qc.measure(i, i)
        
        return qc

    def simulate(self, steps):
        """Run the quantum walk simulation."""
        circuit = self.create_circuit(steps)
        
        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend, shots=20000)
        result = job.result()
        counts = result.get_counts(circuit)
        
        probabilities = {(x, y): 0.0 for x in range(self.grid_size) 
                       for y in range(self.grid_size)}
        
        total_shots = sum(counts.values())
        for state, count in counts.items():
            x_bits = state[:self.num_position_qubits]
            y_bits = state[self.num_position_qubits:2*self.num_position_qubits]
            
            x = int(x_bits, 2) % self.grid_size
            y = int(y_bits, 2) % self.grid_size
            
            probabilities[(x, y)] = count / total_shots
        
        # Check if all positions have been visited
        if all(prob > 0 for prob in probabilities.values()) and not self.is_complete:
            self.is_complete = True
            self.completion_step = steps
        
        return probabilities

class ClassicalWalk:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.start_pos = (0, 0)  # Starting at origin
        self.is_complete = False
        self.completion_step = None

    def simulate(self, steps, num_walkers=20000):
        """Simulate classical random walk."""
        # Initialize all walkers at start_pos
        positions = np.zeros((num_walkers, 2), dtype=int)  # All walkers start at (0,0)
        
        # Possible movements: up, right, down, left
        possible_moves = np.array([
            [-1, 0],  # up
            [0, 1],   # right
            [1, 0],   # down
            [0, -1],  # left
        ])
        
        for step in range(steps):
            # Each walker independently chooses a random direction
            random_indices = np.random.randint(0, 4, size=num_walkers)
            moves = possible_moves[random_indices]
            
            # Calculate new positions
            new_positions = positions + moves
            
            # Apply boundary conditions: if walker would go out of bounds,
            # it stays in place instead of wrapping around
            valid_moves = (new_positions >= 0) & (new_positions < self.grid_size)
            valid_steps = valid_moves[:, 0] & valid_moves[:, 1]
            
            # Update only valid moves
            positions[valid_steps] = new_positions[valid_steps]
            
            # Check if all positions have been visited at this step
            if not self.is_complete:
                unique_positions = set(map(tuple, positions))
                all_positions = {(x, y) for x in range(self.grid_size) 
                               for y in range(self.grid_size)}
                if unique_positions >= all_positions:
                    self.is_complete = True
                    self.completion_step = step + 1
        
        # Count walkers at each position
        probabilities = {(x, y): 0.0 for x in range(self.grid_size) 
                       for y in range(self.grid_size)}
        
        # Use numpy's unique function with axis parameter for 2D positions
        unique_positions, counts = np.unique(positions, axis=0, return_counts=True)
        for pos, count in zip(unique_positions, counts):
            probabilities[tuple(pos)] = count / num_walkers
        
        return probabilities