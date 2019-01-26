

async function main() {
    var iForm = document.getElementById('train');
    var pForm = document.getElementById('predict');
    var rDiv = document.getElementById('results');

    if (getParameterByName("i1")) {     //first submit with training data
        iForm.style.visibility = 'hidden';
        pForm.style.visibility = 'hidden';
        rDiv.style.visibility = "hidden";

        console.log("training please wait!!");

        let i1 = getParameterByName("i1");
        let i2 = getParameterByName("i2");
        let i3 = getParameterByName("i3");
        let i4 = getParameterByName("i4");
        let i5 = getParameterByName("i5");
        let i6 = getParameterByName("i6");

        let o1 = getParameterByName("o1");
        let o2 = getParameterByName("o2");
        let o3 = getParameterByName("o3");
        let o4 = getParameterByName("o4");
        let o5 = getParameterByName("o5");
        let o6 = getParameterByName("o6");

        await run([i1, i2, i3, i4, i5, i6], [o1, o2, o3, o4, o5, o6]);

        console.log("training complete!!");

        pForm.style.visibility = 'visible';

    } else if (getParameterByName("p1")) {    //2nd submit with prediction queries

        let restart = "<br/><br/><a href='.'>Restart?</a>";


 let p1 = getParameterByName("p1");


        iForm.style.visibility = 'hidden';
        pForm.style.visibility = 'hidden';
        rDiv.style.visibility = "visible";
        let model = await tf.loadModel('localstorage://my-model-1');

          document.getElementById('results').innerHTML = p1 + " = "+
               model.predict(tf.tensor2d([p1], [1, 1])).dataSync() + restart;

        
    } else {  //first load, no submits!
        iForm.style.visibility = 'visible';
        pForm.style.visibility = 'hidden';
        rDiv.style.visibility = "hidden";
    }

}

main();


async function run(i,o) {
    // Create a simple model.
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
    }));

    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });

    // Generate some synthetic data for training. (y = 2x - 1)
    const xs = tf.tensor2d(i, [6, 1]);
    const ys = tf.tensor2d(o, [6, 1]);

    // Train the model using the data.
    await model.fit(xs, ys, {
        epochs: 500
    });

    await model.save('localstorage://my-model-1');

    return true;
    // Use the model to do inference on a data point the model hasn't seen.
    // Should print approximately 39.
  
}






















function getParameterByName(name, url) {
    if (!url) url = window.location.href;
    name = name.replace(/[\[\]]/g, '\\$&');
    var regex = new RegExp('[?&]' + name + '(=([^&#]*)|&|#|$)'),
        results = regex.exec(url);
    if (!results) return null;
    if (!results[2]) return '';
    return parseInt(decodeURIComponent(results[2].replace(/\+/g, ' ')));
}