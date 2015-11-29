import Cocoa

// MARK: Extensions

extension MutableCollectionType where Index == Int {

    mutating func shuffleInPlace() {

        if self.count < 2 {
            
            return
            
        }
        
        for i in 0..<(self.count-1) {
            
            let j = Int(arc4random_uniform(UInt32(self.count - i))) + i
            guard i != j else {
            
                continue
                
            }
            swap(&self[i], &self[j])
            
        }
        
    }
    
}

// MARK: Neural Network

class Network {
    
    var biases: [[Double]] = []
    var weights: [[[Double]]] = []
    let layers: [Int]
    
    init(layers: [Int], biases: [[Double]]?, weights: [[[Double]]]?) {
        
        assert(layers.count != 0)
        self.layers = layers
        
        for (index, neuronCount) in layers.enumerate() {
            
            if index == 0 {
                
                continue
                
            }
            
            var layerBiases: [Double] = []
            for _ in 0..<neuronCount {
                
                if let biases = biases {
                    
                    layerBiases.append(biases[index - 1][layerBiases.count])
                    
                } else {
                    
                    layerBiases.append(Math.randomDouble())
                    
                }
                
            }
            self.biases.append(layerBiases)
            
            var layerWeights: [[Double]] = []
            for _ in 0..<neuronCount {
                
                var neuronWeights: [Double] = []
                for _ in 0..<layers[index - 1] {
                    
                    if let weights = weights {
                        
                        neuronWeights.append(weights[index - 1][layerWeights.count][neuronWeights.count])
                        
                    } else {
                        
                        neuronWeights.append(Math.randomDouble())
                        
                    }
                    
                }
                layerWeights.append(neuronWeights)
                
            }
            self.weights.append(layerWeights)
            
        }
        
    }
    
    private func calculateNeuronZ(input: [Double], weights: [Double], bias: Double) -> Double {
     
        return Math.dotProduct(input, weights) + bias
        
    }
    
    private func calculateNeuronA(neuronZ: Double) -> Double {
        
        return Math.sigmoid(neuronZ)
        
    }
    
    func feedforward(var values: [Double]) -> [Double] {
        
        // process the input values
        
        let inputLayer = self.layers.first!
        assert(inputLayer == values.count)
        
        // process the layers
        
        for (index, layer) in self.layers.enumerate() {
            
            if index == 0 {
                
                continue
                
            }
            
            var newValues: [Double] = []
            for neuronIndex in 0..<layer {
                
                newValues.append(self.calculateNeuronA(self.calculateNeuronZ(values, weights: self.weights[index - 1][neuronIndex], bias: self.biases[index - 1][neuronIndex])))
                
            }
            values = newValues
            
        }
        
        return values;
        
    }
    
    typealias InputSample = (input: [Double], desiredResult: [Double])
    func train(var input: [InputSample], stochasticSampleSize: Int, learningRate: Double) {
        
        // we will be implementing stochastic gradient descent, hence we must shuffle the array and sort it into samples
        
        input.shuffleInPlace()
        
        var inputSamples: [[InputSample]] = []
        for var i = 0; i < input.count; i += stochasticSampleSize {
            
            var sample: [InputSample] = []
            for j in 0..<stochasticSampleSize {
                
                if i + j == input.count {
                    
                    break;
                    
                }
                sample.append(input[i + j])
                
            }
            inputSamples.append(sample)
            
        }
        
        // now we can perform gradient descent on each sample
        
        for (index, sample) in inputSamples.enumerate() {
            
            let sampleCount = index * sample.count
            if sampleCount % 1000 == 0 {
                
                print("Processed \(sampleCount) samples...")
                
            }
            
            self.gradientDescent(sample, learningRate: learningRate)
            
        }
        
    }
    
    private func gradientDescent(sample: [InputSample], learningRate: Double) {
        
        var biasChanges = self.biases.map { $0.map { _ in 0.0 } }
        var weightChanges = self.weights.map { $0.map { $0.map { _ in 0.0 } } }
        
        for (input, desiredResult) in sample {
            
            let result = self.backpropagate(input, desiredResult: desiredResult)
            for i in 0..<biasChanges.count {
                
                for j in 0..<biasChanges[i].count {
                    
                    biasChanges[i][j] = biasChanges[i][j] + result.biasChanges[i][j]
                    
                }
                
            }
            for i in 0..<weightChanges.count {
                
                for j in 0..<weightChanges[i].count {
                    
                    for k in 0..<weightChanges[i][j].count {
                        
                        weightChanges[i][j][k] = weightChanges[i][j][k] + result.weightChanges[i][j][k]
                        
                    }
                    
                }
                
            }
            
        }
        
        for i in 0..<biasChanges.count {
            
            for j in 0..<biasChanges[i].count {
                
                self.biases[i][j] = self.biases[i][j] - (Double(learningRate) / Double(sample.count)) * biasChanges[i][j]
                
            }
            
        }
        for i in 0..<weightChanges.count {
            
            for j in 0..<weightChanges[i].count {
                
                for k in 0..<weightChanges[i][j].count {
                    
                    self.weights[i][j][k] = self.weights[i][j][k] - (Double(learningRate) / Double(sample.count)) * weightChanges[i][j][k]
                    
                }
                
            }
            
        }
        
    }
    
    typealias BackpropagateResult = (weightChanges:[[[Double]]], biasChanges: [[Double]])
    private func backpropagate(input: [Double], desiredResult: [Double]) -> BackpropagateResult {
        
        var biasChanges = self.biases.map { $0.map { _ in 0.0 } }
        var weightChanges = self.weights.map { $0.map { $0.map { _ in 0.0 } } }
        
        // feedforward and collect the activations/z values
        
        var currentLayerActivations = input
        var activationsList = [input]
        var zList: [[Double]] = []
        
        for (index, layer) in self.layers.enumerate() {
            
            if index == 0 {
                
                continue
                
            }
            
            var zLayer: [Double] = []
            var activationsLayer: [Double] = []
            
            for neuronIndex in 0..<layer {
                
                let z = self.calculateNeuronZ(currentLayerActivations, weights: self.weights[index - 1][neuronIndex], bias: self.biases[index - 1][neuronIndex])
                zLayer.append(z)
                
                let a = self.calculateNeuronA(z)
                activationsLayer.append(a)
                
            }
            
            currentLayerActivations = activationsLayer
            activationsList.append(activationsLayer)
            zList.append(zLayer)
            
        }
        
        // first we can backpropagate the output layer weights
        
        var error_outDerivOutputLayer: [Double] = []
        var out_netDerivOutputLayer: [Double] = []
        
        for (neuronIndex, result) in activationsList[activationsList.count - 1].enumerate() {
            
            let desiredResult = desiredResult[neuronIndex]
            let error_outDeriv = -(desiredResult - result)
            let out_netDeriv = Math.sigmoidPrime(result)
            
            error_outDerivOutputLayer.append(error_outDeriv)
            out_netDerivOutputLayer.append(out_netDeriv)
            
            let biasAdjustment = error_outDeriv * out_netDeriv
            biasChanges[biasChanges.count - 1][neuronIndex] = biasAdjustment
            
            for (weightIndex, _) in self.weights[self.weights.count - 1][neuronIndex].enumerate() {
                
                let net_weightDeriv = activationsList[activationsList.count - 2][weightIndex]
                let weightAdjustment = error_outDeriv * out_netDeriv * net_weightDeriv
                weightChanges[weightChanges.count - 1][neuronIndex][weightIndex] = weightAdjustment
                
            }
            
        }
        
        // now we can backpropagate the hidden layers
        
        for layerIndex in (activationsList.count - 2).stride(to: 0, by: -1) {
            
            for (neuronIndex, result) in activationsList[layerIndex].enumerate() {
                
                // the error_weightDeriv for each weight is equal to error_outDeriv * out_netDeriv * net_weightDeriv
                // unlike in the output layer, error_outDeriv is error(0)_outDeriv * ... * error(n)_outDeriv for all next-layer nodes that it affects
                
                var error_outDeriv = 0.0
                for (error_outIndex, _) in activationsList[layerIndex + 1].enumerate() {
                    
                    let outputerror_netDeriv = error_outDerivOutputLayer[error_outIndex] * out_netDerivOutputLayer[error_outIndex]
                    let outputnet_outDeriv = self.weights[layerIndex][error_outIndex][neuronIndex]
                    error_outDeriv += outputerror_netDeriv * outputnet_outDeriv
                    
                }
                let out_netDeriv = Math.sigmoidPrime(result)
                
                let biasAdjustment = error_outDeriv * out_netDeriv
                biasChanges[layerIndex - 1][neuronIndex] = biasAdjustment
                
                for (weightIndex, _) in self.weights[layerIndex - 1][neuronIndex].enumerate() {
                    
                    let net_weightDeriv = activationsList[layerIndex - 1][weightIndex]
                    let weightAdjustment = error_outDeriv * out_netDeriv * net_weightDeriv
                    weightChanges[layerIndex - 1][neuronIndex][weightIndex] = weightAdjustment
                    
                }
                
            }
            
        }
        
        return (weightChanges: weightChanges, biasChanges: biasChanges)
        
    }
    
}

class Math {
    
    static func dotProduct(a: [Double], _ b: [Double]) -> Double {
        
        assert(a.count == b.count)
        
        var result = 0.0
        for i in 0..<a.count {
            
            result += a[i] * b[i]
            
        }
        return result
        
    }
    
    static func randomDouble() -> Double {
        
        return Double(arc4random()) / Double(UINT32_MAX)
        
    }
    
    static func sigmoid(z: Double) -> Double {
        
        return 1.0 / (1.0 + pow(M_E, -z))
        
    }
    
    static func sigmoidPrime(z: Double) -> Double {
        
        return self.sigmoid(z) * (1 - self.sigmoid(z))
        
    }
    
}

//let initialWeights = [[[0.15, 0.20], [0.25, 0.30]], [[0.40, 0.45], [0.50, 0.55]]]
//let initialBiases = [[0.35, 0.35], [0.60, 0.60]]
/*let network = Network(layers: [2, 4, 2], biases: nil, weights: nil)
print(network.feedforward([0.05, 0.10]))
var trainingData: [Network.InputSample] = []
for _ in 0..<10000 {
    
    trainingData.append((input: [0.05, 0.10], desiredResult: [0.01, 0.99]))
    
}
network.train(trainingData, stochasticSampleSize: 10, learningRate: 0.5)
print(network.feedforward([0.05, 0.10]))*/

let network = Network(layers: [1, 4, 2], biases: nil, weights: nil)
print(network.feedforward([60]))
print(network.feedforward([45]))
print(network.feedforward([13]))
print(network.feedforward([21]))
print(network.feedforward([97]))
print(network.feedforward([4]))
var trainingData: [Network.InputSample] = []
for i in 0..<100000 {
    
    trainingData.append((input: [Double(i % 100)], desiredResult: [0.25, 0.75]))
    
}
network.train(trainingData, stochasticSampleSize: 10, learningRate: 0.5)
print(network.feedforward([60]))
print(network.feedforward([45]))
print(network.feedforward([13]))
print(network.feedforward([21]))
print(network.feedforward([97]))
print(network.feedforward([4]))
