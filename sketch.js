/* 
function setup() {
  createCanvas(400, 400);
}

function draw() {
  background(220);
} 
*/

console.log('Hello TensorFlow');

function createModel(n_hidden=100) {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

 // Add hidden layer
  model.add(tf.layers.dense({units: n_hidden, activation: 'ReLU'}));
  
  // Add 2nd hidden layer
  model.add(tf.layers.dense({units: 100, activation: 'ReLU'}));

  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}

function getData(N_train=50, N_test=50) {
  // Generate dataset using polynomial function
  const N = N_train + N_test   // number of points in dataset
  x = tf.randomUniform(shape=[N,1], minval=-2, maxval=2, seed=0);
  x.print();
  y= x.add(0.8).mul(0.5).mul(x.add(1.8)).mul(x.sub(0.2)).mul(x.sub(0.3)).mul(x.sub(1.9)).add(1)
  //0.5*(x+0.8)*(x+1.8)*(x-0.2)*(x-0.3)*(x-1.9)+1;
  y.print();

  // stdDev = sqrt(var) -> sqrt(0.05) = 0.2236
  y_noisy = y.add(tf.randomNormal(shape=[N,1], mean=0, stdDev=0.2236));
  y_noisy.print();
  
  // Split into training and testing
  const [x_train, x_test] = tf.split(x, [N_train, N_test], 0);
  const [y_train, y_test] = tf.split(y, [N_train, N_test], 0);
  const [y_noisy_train, y_noisy_test] = tf.split(y_noisy, [N_train, N_test], 0);
  
  //console.log("Training set:");
  //y_test.print();
                  
  return {
      inputs_train: x_train,
      labels_train: y_train,
      labels_noisy_train: y_noisy_train,
      inputs_test: x_test,
      labels_test: y_test,
      labels_noisy_test: y_noisy_test
    }
}


async function run() {
  // Load and plot the original input data that we are going to train on.
  
  // Select dataset size to be generated
  const N_train = 50;
  const N_test = 50;
  const data = await getData(N_train, N_test);
  
  n_epochs = 100;
  n_epochs_overfit = 100;
  
  console.log("Training")
  data.inputs_train.print();
 
  const names = ["Inputs", "Outputs"];
        
  data.labels_train.print();

    const xVals_train = await data.inputs_train.array();
    const xVals_test = await data.inputs_test.array();
    const yVals_train = await data.labels_train.array();
    const yVals_noisy_train = await data.labels_noisy_train.array();
    const yVals_test = await data.labels_test.array();
    const yVals_noisy_test = await data.labels_noisy_test.array();

    const data1 = xVals_train.map((x, i) => ({ x, y: yVals_train[i] }));
    const data2 = xVals_test.map((x, i) => ({ x, y: yVals_test[i] }));
    const data1_noisy = xVals_train.map((x, i) => ({ x, y: yVals_noisy_train[i] }));
    const data2_noisy = xVals_test.map((x, i) => ({ x, y: yVals_noisy_test[i] }));
  
    tfvis.render.scatterplot(
        { name: 'Clean Dataset'},
        //{ values: data1 },
        {values: [data1, data2], series: ['train', 'test']},
        {
          xLabel: 'X (Input)',
          yLabel: 'Y (Label)',
          height: 300
        }
      );
  
      tfvis.render.scatterplot(
        { name: 'Noisy Dataset'},
        //{ values: data1 },
        {values: [data1_noisy, data2_noisy], series: ['train', 'test']},
        {
          xLabel: 'X (Input)',
          yLabel: 'Y (Label)',
          height: 300
        }
      );

  // More code will be added below
  // Create the model
const model = createModel();
tfvis.show.modelSummary({name: 'Model Summary'}, model);
  // Convert the data to a form we can use for training.
//const tensorData = convertToTensor(data);
//const {inputs, labels} = tensorData;

// Train the model
await trainModel(model, data.inputs_train, data.labels_train, n_epochs);
console.log('Done Training (Clean)');


// Make some predictions using the model and compare them
predictions_train = testModel(model, data.inputs_train);
predictions_test = testModel(model, data.inputs_test);
  
mse_train = tf.losses.meanSquaredError(tf.reshape(data.labels_train,[-1]), predictions_train).dataSync()[0]
mse_test = tf.losses.meanSquaredError(tf.reshape(data.labels_test,[-1]), predictions_test).dataSync()[0]

const pred_train = xVals_train.map((x, i) => ({ x, y: predictions_train[i] }));
const pred_test = xVals_test.map((x, i) => ({ x, y: predictions_test[i] }));
  
tfvis.render.scatterplot(
    {name: `Clean Model Predictions (Train), MSE=${mse_train.toFixed(3).toString()}`},
    {values: pred_train, series: ['train']},
    {
      xLabel: 'X (Input)',
      yLabel: 'Y (Label)',
      height: 300
    }
  );
tfvis.render.scatterplot(
    {name: `Clean Model Predictions (Test), MSE=${mse_test.toFixed(3).toString()}`},
    {values: pred_test, series: ['test']},
    {
      xLabel: 'X (Input)',
      yLabel: 'Y (Label)',
      height: 300
    }
  );
  
// Create SECOND model to train with noisy data (BEST-FIT)
const model_noisy = createModel();
  
// Train the model
await trainModel(model_noisy, data.inputs_train, data.labels_noisy_train, n_epochs);
console.log('Done Training (Noisy)');

// Make some predictions using the model and compare them
predictions_noisy_train = testModel(model_noisy, data.inputs_train);
predictions_noisy_test = testModel(model_noisy, data.inputs_test);

const pred_noisy_train = xVals_train.map((x, i) => ({ x, y: predictions_noisy_train[i] }));
const pred_noisy_test = xVals_test.map((x, i) => ({ x, y: predictions_noisy_test[i] }));

mse_noisy_train = tf.losses.meanSquaredError(tf.reshape(data.labels_noisy_train,[-1]), predictions_noisy_train).dataSync()[0]
mse_noisy_test = tf.losses.meanSquaredError(tf.reshape(data.labels_noisy_test,[-1]), predictions_noisy_test).dataSync()[0]
  
tfvis.render.scatterplot(
    {name: `Noisy Model Predictions (Train), MSE=${mse_noisy_train.toFixed(3).toString()}`},
    {values: pred_noisy_train, series: ['train']},
    {
      xLabel: 'X (Input)',
      yLabel: 'Y (Label)',
      height: 300
    }
  );
  
tfvis.render.scatterplot(
    {name: `Noisy Model Predictions (Test), MSE=${mse_noisy_test.toFixed(3).toString()}`},
    {values: pred_noisy_test, series: ['test']},
    {
      xLabel: 'X (Input)',
      yLabel: 'Y (Label)',
      height: 300
    }
  );
  
  
// Create THIRD model to train with noisy data and OVER-FIT
const model_noisy_overfit = createModel();
  
// Train the model
await trainModel(model_noisy_overfit, data.inputs_train, data.labels_noisy_train, n_epochs_overfit);
console.log('Done Training (Noisy)');

// Make some predictions using the model and compare them
predictions_noisy_overf_train = testModel(model_noisy_overfit, data.inputs_train);
predictions_noisy_overf_test = testModel(model_noisy_overfit, data.inputs_test);

const pred_noisy_overf_train = xVals_train.map((x, i) => ({ x, y: predictions_noisy_overf_train[i] }));
const pred_noisy_overf_test = xVals_test.map((x, i) => ({ x, y: predictions_noisy_overf_test[i] }));

mse_noisy_overf_train = tf.losses.meanSquaredError(tf.reshape(data.labels_noisy_train,[-1]), predictions_noisy_overf_train).dataSync()[0]
mse_noisy_overf_test = tf.losses.meanSquaredError(tf.reshape(data.labels_noisy_test,[-1]), predictions_noisy_overf_test).dataSync()[0]
  
tfvis.render.scatterplot(
    {name: `Noisy Overfit Model Predictions (Train), MSE=${mse_noisy_overf_train.toFixed(3).toString()}`},
    {values: pred_noisy_overf_train, series: ['train']},
    {
      xLabel: 'X (Input)',
      yLabel: 'Y (Label)',
      height: 300
    }
  );
  
tfvis.render.scatterplot(
    {name: `Noisy Overfit Model Predictions (Test), MSE=${mse_noisy_overf_test.toFixed(3).toString()}`},
    {values: pred_noisy_overf_test, series: ['test']},
    {
      xLabel: 'X (Input)',
      yLabel: 'Y (Label)',
      height: 300
    }
  );
  
  
}

document.addEventListener('DOMContentLoaded', run);


async function trainModel(model, inputs, labels, n_epochs) {
  // Prepare the model for training.
  learning_rate = 0.01
  model.compile({
    optimizer: tf.train.adam(learning_rate),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
    
  });

  const batchSize = 32;
  const epochs = n_epochs;

  
  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

function testModel(model, inputData) {
  // Generate predictions for given inputData;
  const predictions = model.predict(inputData);
  
  /*const predictedPoints = Array.from(inputData).map((val, i) => {
    return {x: val, y: predictions.dataSync()[i]}
  }); */
  
  console.log("predictions:");
  console.log(predictions.dataSync());
  return predictions.dataSync();
}



