 const canvas = document.getElementById("canvas");
 const guide = document.getElementById("guide");
 const colorInput = document.getElementById("colorInput");
 const toggleGuide = document.getElementById("toggleGuide");
 const clearButton = document.getElementById("clearButton");
 const drawingContext = canvas.getContext("2d");
 
 const CELL_SIDE_COUNT = 64;
 const cellPixelLength = canvas.width / CELL_SIDE_COUNT;
 const colorHistory = {};
 
 // Set default color
 colorInput.value = "#FF00FF";
 
 // Initialize the canvas background
 drawingContext.fillStyle = "white";
 drawingContext.fillRect(0, 0, canvas.width, canvas.height);
 const img = document.getElementById("inp");
 drawingContext.drawImage(img, 0, 0);
 
 // Setup the guide
 {
   guide.style.width = `${canvas.width}px`;
   guide.style.height = `${canvas.height}px`;
   guide.style.gridTemplateColumns = `repeat(${CELL_SIDE_COUNT}, 1fr)`;
   guide.style.gridTemplateRows = `repeat(${CELL_SIDE_COUNT}, 1fr)`;
 
   [...Array(CELL_SIDE_COUNT ** 2)].forEach(() =>
     guide.insertAdjacentHTML("beforeend", "<div></div>")
   );
 }
 
 function handleCanvasMousedown(e) {
   // Ensure user is using their primary mouse button
   if (e.button !== 0) {
     return;
   }
 
   const canvasBoundingRect = canvas.getBoundingClientRect();
   const x = e.clientX - canvasBoundingRect.left;
   const y = e.clientY - canvasBoundingRect.top;
   const cellX = Math.floor(x / cellPixelLength);
   const cellY = Math.floor(y / cellPixelLength);
   const currentColor = colorHistory[`${cellX}_${cellY}`];
 
   if (e.ctrlKey) {
     if (currentColor) {
       colorInput.value = currentColor;
     }
   } else {
     fillCell(cellX, cellY);
   }
 }
 
 function handleClearButtonClick() {

   drawingContext.fillStyle = "#ffffff";
   drawingContext.fillRect(0, 0, canvas.width, canvas.height);
 }
 
 function handleToggleGuideChange() {
   guide.style.display = toggleGuide.checked ? null : "none";
 }
 
 function fillCell(cellX, cellY) {
   const startX = cellX * cellPixelLength;
   const startY = cellY * cellPixelLength;
 
   drawingContext.fillStyle = colorInput.value;
   drawingContext.fillRect(startX, startY, cellPixelLength, cellPixelLength);
   colorHistory[`${cellX}_${cellY}`] = colorInput.value;
 }
 canvas.onmousedown = function(e){
    canvas.onmousemove = handleCanvasMousedown;
    handleCanvasMousedown();
 }

 canvas.onmouseup = function(e){
    canvas.onmousemove = null;
 }

 canvas.addEventListener("mousedown", handleCanvasMousedown);
 clearButton.addEventListener("click", handleClearButtonClick);
 toggleGuide.addEventListener("change", handleToggleGuideChange);

 function save() {
    var image = new Image();
    var url = document.getElementById('url');
    image.id = "pic";
    image.src = canvas.toDataURL();
    url.value = image.src;

}

function changeLabel(){
    const color = document.querySelector('#colorInput').value;
    if (color == '#ff00ff') {
        document.querySelector('label[for="colorInput"]').innerHTML = 'Mountain';
      }
    if (color == '#ff0000') {
    document.querySelector('label[for="colorInput"]').innerHTML = 'Forest';
    }
    if (color == '#32cd32') {
        document.querySelector('label[for="colorInput"]').innerHTML = 'Grasslands';
      }
    if (color == '#0000ff') {
    document.querySelector('label[for="colorInput"]').innerHTML = 'Deep Waters';
    }
    if (color == '#ffffff') {
        document.querySelector('label[for="colorInput"]').innerHTML = 'Salt Flats';
      }
    if (color == '#00ffff') {
    document.querySelector('label[for="colorInput"]').innerHTML = 'Shallow Waters';
    }
    if (color == '#ffff00') {
        document.querySelector('label[for="colorInput"]').innerHTML = 'Dessert';
      }

  }
  
  window.onload = () => {
    document.querySelector('#colorInput').addEventListener('change', changeLabel);
    changeLabel();
  }

  function onlyOne(checkbox) {
    var checkboxes = document.getElementsByName('check')
    checkboxes.forEach((item) => {
        if (item !== checkbox) item.checked = false
    })
}

