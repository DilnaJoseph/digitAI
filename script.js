
// canvas -------------------------------------------------------------------------------------------------------------------------
const canvas = document.getElementById("drawcanvas");
const ctx = canvas.getContext("2d");
ctx.lineWidth = 12;
ctx.lineCap = "round";
ctx.strokeStyle = "#ffffff";
ctx.shadowColor = "rgba(255, 255, 255, 0.8)";
ctx.shadowBlur = 10;
ctx.shadowOffsetX = 0;
ctx.shadowOffsetY = 0;
let drawing = false;

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return { x: clientX - rect.left, y: clientY - rect.top };
}

function startDraw(e) {
  e.preventDefault();
  drawing = true;
  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
}

function draw(e) {
  if (!drawing) return;
  e.preventDefault();
  const { x, y } = getPos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
}

function endDraw() {
  drawing = false;
}

// mouse events
canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", endDraw);

// touch events
canvas.addEventListener("touchstart", startDraw);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", endDraw);

// CLEAR button
document.querySelector(".clear").addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});


// image upload -------------------------------------------------------------------------------------------------------------------------

const imgInput = document.getElementById("imgInput");

document.querySelector(".imgupload").addEventListener("click", () => {
  imgInput.click(); // opens filepicker
});

imgInput.addEventListener("change", () => {  // runs when imgInput changes
  const file = imgInput.files[0]; // only first file
  if (!file) return;

  const reader = new FileReader(); // reads the file
  reader.onload = (e) => { // onload means run once the file is read
    const img = new Image();
    img.onload = () => { // obtains img dimensions
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const scale = Math.min(canvas.width / img.width, canvas.height / img.height); // to prevent stretching of image
      const w = img.width * scale;
      const h = img.height * scale; 
      // to center image
      const x = (canvas.width - w) / 2;
      const y = (canvas.height - h) / 2;
      ctx.drawImage(img, x, y, w, h);
    };
    img.src = e.target.result;
  };

  reader.readAsDataURL(file); // img file -> data file
});


// generate btn -------------------------------------------------------------------------------------------------------------------------

// MNIST input creation
async function getMnistInput() {
  const off = document.createElement("canvas"); // creates a new canvas to reduce size and draw
  off.width = 28;
  off.height = 28;
  const offCtx = off.getContext("2d"); // allows drawing

  offCtx.drawImage(canvas, 0, 0, 28, 28); // downscale whatever is on the main canvas to 28x28 and draws it on the new canvas

  const imgData = offCtx.getImageData(0, 0, 28, 28);
  const data = imgData.data;
  const pixels = [];

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const gray = (r + g + b) / 3;
    pixels.push(gray / 255);           
  }

  return pixels;                      
}



document.querySelector(".generate").addEventListener("click", async () => {
  const mnistInput = await getMnistInput();
  console.log("MNIST array length:", mnistInput.length); // should be 784
 
});




// retrain btn -------------------------------------------------------------------------------------------------------------------------

const addBtn = document.querySelector(".retrain");
const correctInput = document.getElementById("correctDigit");

const feedbackSamples = [];  

addBtn.addEventListener("click", async () => {
  const label = parseInt(correctInput.value, 10);
  if (Number.isNaN(label) || label < 0 || label > 9) {
    alert("Please enter a digit between 0 and 9.");
    return;
  }

  const pixels = await getMnistInput();  

  feedbackSamples.push({ x: pixels, y: label });

  console.log("Added feedback sample:", feedbackSamples.length);
});



// updating UI -------------------------------------------------------------------------------------------------------------------------

function updatePredictionUI(probs){
  const mainDigitEl = document.getElementById("mainDigit");
  const mainAccEl   = document.getElementById("mainAcc");
  const colLeft     = document.getElementById("col-left");
  const colRight    = document.getElementById("col-right");

  colLeft.innerHTML  = "";
  colRight.innerHTML = "";

  const maxIndex = probs.indexOf(Math.max(...probs));
  const maxPercent = Math.round(probs[maxIndex] * 100);

  mainDigitEl.textContent = maxIndex;
  mainAccEl.textContent   = `Accuracy: ${maxPercent}%`;

  probs.forEach((p, digit) => {
    const percent = Math.round(p * 100);

    const row = document.createElement("div");
    row.className = "pred-row" + (digit === maxIndex ? " active" : "");

    row.innerHTML = `
      <span class="digit">${digit}</span>
      <div class="bar"><div class="fill" style="width:${percent}%;"></div></div>
      <span class="percent">${percent}%</span>
    `;

    if (digit % 2 === 0) colLeft.appendChild(row); 
    else colRight.appendChild(row);             
  });
}

let tfModel = null;

(async () => {
  tfModel = await tf.loadLayersModel("https://storage.googleapis.com/tfjs-models/tfjs/mnist/model.json");
  console.log("TF.js model loaded");
})();

document.querySelector(".generate").addEventListener("click", async () => {
  if (!tfModel) {
    alert("Model is still loading, please wait a moment.");
    return;
  }

  const pixels = await getMnistInput();      

  const input = tf.tensor(pixels, [1, 784]);

  // run model
  const logits = tfModel.predict(input);
  const probsTensor = await logits.data();
  const probs = Array.from(probsTensor);        

  input.dispose();
  logits.dispose();

  updatePredictionUI(probs);
});
