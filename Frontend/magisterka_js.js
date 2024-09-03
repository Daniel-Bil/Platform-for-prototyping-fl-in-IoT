//IMPORTED MODULES FOR READABILITY
import { generateStruct } from './generateStruct.js'; //Function creating all layers from tf.layers
import { generateArchitectureJSON } from './jsonGenerator.js';
//ARRAY STORING VALUES OF CREATED LAYERS
const elementsData = [];
const GotArchitectures = [];

// variables needed for creating line conn3ction
let svgContainer = document.getElementById('svgContainer');
let isDrawing = false;
let startButton = null;
let currentLine = null;

const connections = [];
// Menu on RIGHT click
// Menu that has names of all layers
////////////////////////////////////
const sendButton = document.getElementById("uploadButton");
const downloadButton = document.getElementById("downloadButton");
const deleteButton = document.getElementById("deleteButton");
const updateButton = document.getElementById("updateButton");
const testButton = document.getElementById("testButton");



let curentModel = "Default"

const architectureNameInput = document.getElementsByClassName("ArchitectureNameInput")[0]
const architectureNameCombo = document.getElementsByClassName("ArchitecturesCombo")[0]
///////////////////////////////////
document.onclick = hideMenu;
document.oncontextmenu = rightClick;


function hideMenu() {
    document.getElementById(
        "contextMenu").style.display = "none"
}

function rightClick(e) {
    e.preventDefault();
    if (document.getElementById(
        "contextMenu").style.display == "block")
        hideMenu();
    else {
        let menu = document.getElementById("contextMenu")

        console.log(e.pageX);
        console.log(e.pageY);

        menu.style.left = e.pageX + "px";
        menu.style.top = e.pageY + "px";
        menu.style.display = 'block';
    }
}

// ADDING LAYERS AFTER CLICKING

document.addEventListener('DOMContentLoaded', function() {
    // Select the parent element that contains all the <li> elements
    var contextMenu = document.getElementById('contextMenu');

    // Add event listener to the parent element
    contextMenu.addEventListener('click', function(event) {
        // Check if the clicked element is an <li>
        var target = event.target;
        if (target.tagName === 'LI' && target.childNodes.length<2) {

            // Get the text content of the clicked <li> element
            var textContent = target.textContent.trim();
            console.log('Clicked text content:', textContent);
            // Perform any action with the textContent

            const elemntId = 'element-' + Date.now()

            let newLayer = createLayer(event, elemntId, textContent)

            document.getElementById("list").appendChild(newLayer);
            newLayer.addEventListener('mousedown', mouseDownLayer)

            elementsData.push(generateStruct(textContent, elemntId))
            console.log(elementsData)
            let buttons = newLayer.querySelectorAll('.connector');
            buttons.forEach(button =>{
                button.addEventListener('mousedown', (event) => startLine(event, button));
                button.addEventListener('click', (event) => finishLine(event, button)); // Finish line on button click
            })
        }
    });
});

function createLineElement(x1, y1, x2, y2) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");

    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);

    line.setAttribute("stroke", "white");
    line.setAttribute("stroke-width", "10");
    line.setAttribute("style", "cursor: pointer;");
    line.setAttribute("style", "z-index: 999;");

    svgContainer.appendChild(line);
    return line;
}

function startLine(event, button) {
    if(isDrawing){
        return;
    }
    console.log("startLine")
    isDrawing = true;
    startButton = button;

    const rect = button.getBoundingClientRect();
    const startX = rect.left + rect.width / 2;
    const startY = rect.top + rect.height / 2;

    currentLine = createLineElement(startX, startY, startX, startY);
    currentLine.setAttribute("id", button.parentElement.id)
    updateLine(event);
}

function reconstructStartLine(button) {

    console.log("reconstructStartLine")
    console.log(button)
    const rect = button.getBoundingClientRect();
    const startX = rect.left + rect.width / 2;
    const startY = rect.top + rect.height / 2;

    let reconstructedLine = createLineElement(startX, startY, startX, startY);
    reconstructedLine.setAttribute("id", button.parentElement.id)
    return reconstructedLine
}

function updateLine(event) {
    if (isDrawing && currentLine && startButton) {
        const rect = startButton.getBoundingClientRect();
        const startX = rect.left + rect.width / 2 + window.scrollX;;
        const startY = rect.top + rect.height / 2 + window.scrollY;;

        console.log("rectangle X Y = ", startX, startY)

        const endX = event.pageX;
        const endY = event.pageY;

        currentLine.setAttribute("x1", startX);
        currentLine.setAttribute("y1", startY);
        currentLine.setAttribute("x2", endX);
        currentLine.setAttribute("y2", endY);
    }
}

function stopLine(event) {
    if (isDrawing && currentLine && !event.target.classList.contains("connector")) {
        console.log("stopLine")
        svgContainer.removeChild(currentLine)
        // const lines = svgContainer.querySelectorAll("line")
        // console.log(lines)
        // const index = lines.findIndex(data => data.id === currentLine.id);
        // if (index !== -1) {
        //     svgContainer.splice(index, 1); // Remove the object from the array
        // }

        isDrawing = false;
        currentLine = null;
        startButton = null;
        console.log("connections = ", connections)
    }
}

document.addEventListener("click", (e)=>{
    stopLine(e)
})

function finishLine(event, button) {
    if (isDrawing && currentLine && startButton) {
        // const connection = connections.find(conn => conn.line === lineElement);

        if (!(button===startButton) && !(button.parentElement.id===startButton.parentElement.id) && !(button.classList.contains(startButton.classList[0]))) {

            // connection.endButton = button;
            // Finalize line position with the end button
            if(button.classList.contains("inputButton"))
            {
                connections.push({start: startButton.parentElement.id, end: button.parentElement.id, line: currentLine, startButton:startButton,endButton:button})
                const rect = button.getBoundingClientRect();
                const endX = rect.left + rect.width / 2 + window.scrollX;;
                const endY = rect.top + rect.height / 2 + window.scrollY;;

                currentLine.setAttribute("x2", endX);
                currentLine.setAttribute("y2", endY);
            }
            else{
                currentLine.setAttribute("id", button.parentElement.id)
                connections.push({start: button.parentElement.id, end: startButton.parentElement.id, line: currentLine, startButton:button,endButton:startButton})
                const rect = button.getBoundingClientRect();
                const endX = rect.left + rect.width / 2 + window.scrollX;;
                const endY = rect.top + rect.height / 2 + window.scrollY;;
                let newx2 = currentLine.getAttribute("x1"), newy2 = currentLine.getAttribute("y1")
                currentLine.setAttribute("x1", endX);
                currentLine.setAttribute("y1", endY);

                currentLine.setAttribute("x2", newx2);
                currentLine.setAttribute("y2", newy2);
            }

            console.log("connections = ",connections)

            currentLine.setAttribute("class", "line");
            currentLine.addEventListener("click", (e)=>{
                delLine(e)
            })

            isDrawing = false;
            currentLine = null;
            startButton = null;
        }

    }
}

function reconstructFinishLine(startButton, endButton, reconstructedLine) {

        if (!(endButton===startButton) && !(endButton.parentElement.id===startButton.parentElement.id) && !(endButton.classList.contains(startButton.classList[0]))) {

            // connection.endButton = button;
            // Finalize line position with the end button
            if(endButton.classList.contains("inputButton"))
            {
                connections.push({start: startButton.parentElement.id, end: endButton.parentElement.id, line: reconstructedLine, startButton:startButton,endButton:endButton})
                const rect = endButton.getBoundingClientRect();
                const endX = rect.left + rect.width / 2 + window.scrollX;;
                const endY = rect.top + rect.height / 2 + window.scrollY;;

                reconstructedLine.setAttribute("x2", endX);
                reconstructedLine.setAttribute("y2", endY);
            }
            else{
                reconstructedLine.setAttribute("id", button.parentElement.id)
                connections.push({start: endButton.parentElement.id, end: startButton.parentElement.id, line: reconstructedLine, startButton:endButton,endButton:startButton})
                const rect = endButton.getBoundingClientRect();
                const endX = rect.left + rect.width / 2 + window.scrollX;;
                const endY = rect.top + rect.height / 2 + window.scrollY;;
                let newx2 = reconstructedLine.getAttribute("x1"), newy2 = reconstructedLine.getAttribute("y1")
                reconstructedLine.setAttribute("x1", endX);
                reconstructedLine.setAttribute("y1", endY);

                reconstructedLine.setAttribute("x2", newx2);
                reconstructedLine.setAttribute("y2", newy2);
            }

            console.log("connections = ",connections)

            reconstructedLine.setAttribute("class", "line");
            reconstructedLine.addEventListener("click", (e)=>{
                delLine(e)
            })
        }

}


document.addEventListener('mousemove', (e) => {
    if (isDrawing && currentLine) {
        currentLine.setAttribute("x2", e.pageX);
        currentLine.setAttribute("y2", e.pageY);
    }
});


function createLayer(event, id, layerName){
    let newLayer = document.createElement('div');
    newLayer.id = id
    newLayer.classList.add('layerType');
    newLayer.classList.add('glass');
    newLayer.classList.add('notChosen');
    newLayer.innerHTML = `
    <button class="inputButton connector glass"></button>
    <label class="layerName">${layerName}</label>
    <span class="close">X</span>

    <button class="outputButton connector glass"></button>
    `

    newLayer.style.left = `${event.pageX-50}px`;
    newLayer.style.top = `${event.pageY-50}px`;
    return newLayer
}

function reconstructLayer(id, layerName, place){
    let newLayer = document.createElement('div');
    newLayer.id = id
    newLayer.classList.add('layerType');
    newLayer.classList.add('glass');
    newLayer.classList.add('notChosen');
    newLayer.innerHTML = `
    <button class="inputButton connector glass"></button>
    <label class="layerName">${layerName}</label>
    <span class="close">X</span>

    <button class="outputButton connector glass"></button>
    `

    newLayer.style.left = `${place.x}px`;
    newLayer.style.top = `${place.y}px`;
    return newLayer
}

function reconstructMoreThanLayer(existingLayerInfo, x, y){
    console.log(existingLayerInfo)
    let newLayer = reconstructLayer(existingLayerInfo.id, existingLayerInfo.name, {x:x, y:y})

    document.getElementById("list").appendChild(newLayer);
    newLayer.addEventListener('mousedown', mouseDownLayer)

    elementsData.push(existingLayerInfo)
    console.log(elementsData)
    let buttons = newLayer.querySelectorAll('.connector');
    buttons.forEach(button =>{
        button.addEventListener('mousedown', (event) => startLine(event, button));
        button.addEventListener('click', (event) => finishLine(event, button)); // Finish line on button click
    })
}

//////////////////////////////////////////////////////////////////////////////////
// MOVING LAYER ELEMENTS/////////////////////
let newX = 0, newY = 0, startX =0, startY =0;

function mouseDownLayer(e){
    startX = e.pageX
    startY = e.pageY

    document.addEventListener('mousemove', mouseMoveLayer)
    document.addEventListener('mouseup', mouseUpLayer)


    let layers = document.getElementsByClassName("layerType")
    for (let i = 0; i < layers.length; i++) {
        const layer = layers[i];
        if (layer.classList.contains('chosen')){
            layer.classList.remove("chosen")
            if(!layer.classList.contains('notChosen')){
                layer.classList.add("notChosen")
            }
        }
    }
    if (e.target.classList.contains('notChosen')){
        e.target.classList.remove("notChosen")
        e.target.classList.add("chosen")
        console.log("chosen id = ", e.target.id)
    }

    document.getElementById("hiddenMenuDataList").innerHTML = "";

    if (e.target.classList.contains('chosen')){
        const index = elementsData.findIndex(data => data.id === e.target.id);
        for (const key in elementsData[index]) {
            if (elementsData[index].hasOwnProperty(key)) {
                if (!(key=="id") && !(key=="name") && !(key=="kwargs")){
                    // console.log(`${key}`)
                    if (elementsData[index][key].type=="string" || elementsData[index][key].type=="boolean"){
                        let newEl = generateSelect(e.target.id,`${key}`,elementsData[index][key].type ,elementsData[index][key])
                        document.getElementsByClassName("hiddenMenuDataList")[0].appendChild(newEl)
                    }

                    if(elementsData[index][key].type=="float" || elementsData[index][key].type=="int"){
                        let newEl = generateInput(e.target.id, `${key}`,elementsData[index][key].type ,elementsData[index][key])
                        document.getElementsByClassName("hiddenMenuDataList")[0].appendChild(newEl)
                    }

                    if(elementsData[index][key].type=="tuple"){
                        let newEl = generateTuple(e.target.id, `${key}`,elementsData[index][key].type ,elementsData[index][key])
                        document.getElementsByClassName("hiddenMenuDataList")[0].appendChild(newEl)
                    }


                }

            }
        }
    }
}

function mouseMoveLayer(e){
    if(e.target.classList.contains('layerType')){
        newX = startX - e.pageX
        newY = startY - e.pageY

        startX = e.pageX
        startY = e.pageY
        e.target.style.top = (e.target.offsetTop - newY) + "px"
        e.target.style.left = (e.target.offsetLeft - newX) + "px"
        updateLinePosition(e)


    }
}
function mouseUpLayer(e){
    document.removeEventListener("mousemove", mouseMoveLayer)
}

function delLine(event){
    const index = connections.findIndex(data => data.line.id === event.target.id);

    if (index !== -1) {
        svgContainer.removeChild(event.target)
        connections.splice(index, 1); // Remove the object from the array
    }

}

function delLine2(id){
    console.log("delLine2")
    for(let i =connections.length-1; i>=0;i--){
        if(connections[i].start==id){
            svgContainer.removeChild(connections[i].line)
            connections.splice(i, 1); // Remove the object from the array
            continue
        }

        if(connections[i].end==id){
            svgContainer.removeChild(connections[i].line)
            connections.splice(i, 1); // Remove the object from the array
            continue
        }
    }

}


function updateLinePosition(event){
        console.log("updateLinePosition")
        const startIndex = connections.findIndex(data => data.start === event.target.id);
        const endIndex = connections.findIndex(data => data.end === event.target.id);

        if (startIndex !==-1){
            const rect = connections[startIndex].startButton.getBoundingClientRect();
            const endX = rect.left + rect.width / 2 + window.scrollX;
            const endY = rect.top + rect.height / 2 + window.scrollY;

            for(let i =0; i<connections.length;i++){
                if(connections[i].start==event.target.id){
                    const line = connections[i].line
                    line.setAttribute("x1", endX);
                    line.setAttribute("y1", endY);
                }
            }

        }

        if (endIndex !==-1){
            const rect = connections[endIndex].endButton.getBoundingClientRect();
            const endX = rect.left + rect.width / 2 + window.scrollX;
            const endY = rect.top + rect.height / 2 + window.scrollY;
            for(let i =0; i<connections.length;i++){
                if(connections[i].end==event.target.id){
                    const line = connections[i].line
                    line.setAttribute("x2", endX);
                    line.setAttribute("y2", endY);
                }
            }
        }
}

///// DELETING
//FIX NEEEEDEDEJhdkjgsjkhgjklshfg
document.addEventListener('click', (event) =>{
    if(event.target.classList.contains('close')){
        const elementId = event.target.parentNode.id;
        const index = elementsData.findIndex(data => data.id === elementId);

        if (connections.length>0){
            delLine2(elementId)
        }

        event.target.parentNode.remove();
        if (index !== -1) {
            elementsData.splice(index, 1); // Remove the object from the array
        }

    }
})
/////////////////////////////////////////// tuple buttons
function addTuple(event, id, labelName){
    let parent = event.target.parentElement;

    let customInput = document.createElement('input');
        customInput.type = "number"
        customInput.classList.add("tupleInput")
        customInput.value = 1

    parent.insertBefore(customInput, event.target)
    updateStructTuple(event, id, labelName)
}
function deleteTuple(event, id, labelName){
    let parent = event.target.parentElement;


    const inputs = parent.querySelectorAll('input');
    if (inputs.length > 1) {
        const lastInput = inputs[inputs.length - 1];
        parent.removeChild(lastInput);
    }

    updateStructTuple(event, id, labelName)
}

/////////////////////////////////////////// UPDATE           Structs on hidden menu
function updateStructString(event, id, labelName) {
    const index = elementsData.findIndex(data => data.id === id);
        for (const key in elementsData[index]) {
            if (elementsData[index].hasOwnProperty(key)) {
                if (key==labelName){
                    console.log("updateStructString")
                    // console.log(elementsData[index].key)
                    // console.log(elementsData[index])
                    elementsData[index][key].default = event.target.value
                    console.log(elementsData)
                    console.log(event.target.value)
                }

            }
        }
}

function updateStructTuple(event, id, labelName) {
    const index = elementsData.findIndex(data => data.id === id);
        for (const key in elementsData[index]) {
            if (elementsData[index].hasOwnProperty(key)) {
                if (key==labelName){
                    console.log("updateStructTuple")
                    let parent = event.target.parentElement;
                    let childInputs = parent.querySelectorAll('input');

                    const valuesList = [];
                    console.log(valuesList)
                    childInputs.forEach(input => {
                        valuesList.push(input.value);
                    });
                    console.log(valuesList)
                    elementsData[index][key].default = valuesList

                    console.log(elementsData)
                }

            }
        }
}

function updateStructNumber(event, id, labelName){
    const index = elementsData.findIndex(data => data.id === id);
        for (const key in elementsData[index]) {
            if (elementsData[index].hasOwnProperty(key)) {
                if (key==labelName){
                    console.log("updateStructNumber")

                    elementsData[index][key].default = event.target.value
                    console.log(elementsData)

                }

            }
        }
}
/////////////////////////////////////// GENERATE
function generateKwargs(elementId, label, type, restOfValues){

}

function generateTuple(elementId, label, type, restOfValues){
    let dataDiv = document.createElement('div');

    let customButton1 = document.createElement('button');
    customButton1.classList.add("tupleButton")
    customButton1.classList.add("DeletorTuple")
    customButton1.textContent="-"

    let customButton2 = document.createElement('button');
    customButton2.classList.add("tupleButton")
    customButton2.classList.add("GeneratorTuple")
    customButton2.textContent="+"
    customButton1.id = elementId
    customButton2.id = elementId

    dataDiv.classList.add('hiddenMenuData');

    dataDiv.innerHTML = `<label>${label}</label>`

    dataDiv.appendChild(customButton1)

    customButton1.addEventListener("click", function(event) {
        deleteTuple(event, elementId, label)
    })

    customButton2.addEventListener("click", function(event) {
        addTuple(event, elementId, label)
    })

    console.log("len = ",restOfValues.default.length)
    for(let i=0; i<restOfValues.default.length; i++){
        let customInput = document.createElement('input');
        customInput.type = "number"
        customInput.classList.add("tupleInput")
        customInput.value = restOfValues.default[i]
        customInput.addEventListener("change", function(event) {
            updateStructTuple(event, elementId, label)
        })
        dataDiv.appendChild(customInput)
    }

    dataDiv.appendChild(customButton2)
    return dataDiv
}

function generateSelect(elementId, label, type, restOfValues){
    let dataDiv = document.createElement('div');
    let customSelect = document.createElement('select');
    customSelect.id = elementId
    dataDiv.classList.add('hiddenMenuData');

    dataDiv.innerHTML = `
            <label>${label}</label>
            `
    for (let i=0; i< restOfValues.values.length;i++)
    {
        let customOption = document.createElement("option");
        customOption.text = restOfValues.values[i];
        customOption.value = restOfValues.values[i];
        if(label=="center"){
            console.log("printing")
            console.log(restOfValues.values[i])
            console.log(restOfValues.default)
            console.log(restOfValues.values[i]==restOfValues.default)
        }

        if (restOfValues.values[i]==restOfValues.default){
            customOption.selected = true;
        }

        customSelect.appendChild(customOption)
    }
    dataDiv.appendChild(customSelect)

    customSelect.addEventListener("change", function(event) {
        updateStructString(event, elementId, label)
    })

    return dataDiv
}

function generateInput(elementId, label, type, restOfValues){
    let dataDiv = document.createElement('div');
    dataDiv.classList.add('hiddenMenuData');


    dataDiv.innerHTML = `
            <label>${label}</label>
            `
    let input = document.createElement("input");
    input.type = "number"

    if (type==="float"){
        input.step="0.0001"
    }
    if(restOfValues.default == null){
        input.placeholder = 0
    }
    else{
        input.placeholder = restOfValues.default
    }

    dataDiv.appendChild(input)

    input.addEventListener("change", function(event) {
        updateStructNumber(event, elementId, label)
    })
    return dataDiv
}

function generateSomething(elementId, label, type, restOfValues){

}










///////////////////////////////////////////////// Hiddenmenu
// Hideable Menu on the right - changes to parameters of layers
///////////////////////////////////////////////////////////////
const hiddenButton = document.getElementById("hiddenButton")
hiddenButton.onclick = function(){
    let menu = document.getElementsByClassName("informationMenu")[0]
    if (menu.classList.contains('hidden')){
        menu.classList.remove("hidden")
        menu.classList.add("notHidden")
        menu.style.transform = "translate(-100%)"

        const img = document.querySelector('#hiddenButton img');
        img.style.transform = 'rotate(270deg)';
    }
    else{
        menu.classList.remove("notHidden")
        menu.classList.add("hidden")
        menu.style.transform = "translate(0)"
        const img = document.querySelector('#hiddenButton img');
        img.style.transform = 'rotate(90deg)';
    }
}




// function createJSONArchitecture(){
//     //
//     //
//     //
//     console.log("createJSONArchitecture")
//     let MainJSON = []
//     let MainData = {Architecture: "Sequential"}
//     MainJSON.push(MainData)
//     let layers = document.querySelectorAll(".layerType")
//     if(layers.length>0){
//         for (let layer of layers){
//             const index = elementsData.findIndex(data => data.id === layer.id);
//             // for (const key in elementsData[index]) {
//             //     if (elementsData[index].hasOwnProperty(key)) {
//             //         if (!(key=="id") && !(key=="name") && !(key=="kwargs")){
//             //             console.log(`${key}`)



//             //         }

//             //     }

//             // }
//             MainJSON.push(elementsData[index])



//             console.log("")
//         }
//         let combinedJsonString = JSON.stringify(MainJSON);
//             console.log(combinedJsonString);
//         return combinedJsonString
//     }

// }



////////////////////////////////////////////////// SENDING DATA TEST

var xhr = null;
var getXmlHttpRequestObject = function () {
    if (!xhr) {
        // Create a new XMLHttpRequest object
        xhr = new XMLHttpRequest();
    }
    return xhr;
};


function sendDataCallback() {
    // Check response is ready or not
    if (xhr.readyState == 4 && xhr.status == 201) {
        console.log("Data creation response received!");
    }
    if (xhr.readyState == 4 && xhr.status == 400) {
        alert("Architecture hasnt been send wrong coding contact programmer")
    }
    if (xhr.readyState == 4 && xhr.status == 500) {
        alert("Architecture has been send but wrong backend contact programmer")
    }
}

function sendData() {
    console.log("send data function ");
    let dataToSend = generateArchitectureJSON(elementsData, connections)

    console.log(dataToSend)

    if (!dataToSend) {
        console.log("Data is empty.");
        return;
    }
    const inputShapeInput = document.getElementsByClassName("ArchitectureShapeInput")[0]
    dataToSend[0][0].input_shape = inputShapeInput.value
    console.log(dataToSend)
    let dataToSend2 = {name: architectureNameInput.value, architecture: dataToSend}
    console.log("Sending data:");
    xhr = getXmlHttpRequestObject();
    xhr.onreadystatechange = sendDataCallback;
    // asynchronous requests
    xhr.open("POST", "http://localhost:6969/Architecture", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    // Send the request over the network

    xhr.send(JSON.stringify(dataToSend2));



}


sendButton.onclick = sendData;




///////////////////////////////////////////////////////////////////////////////

function GetDataCallback(){
    // Check response is ready or not
    if (xhr.readyState == 4 && xhr.status == 201) {
        console.log("Data creation response received!");
        let architecturesData = JSON.parse(xhr.response)
        console.log(architecturesData.length)
        architectureNameCombo.innerHTML = ''
        let customOption = document.createElement("option");
            customOption.text = "Default"
            customOption.value = "Default"
            architectureNameCombo.appendChild(customOption)
        for(let i=0;i<architecturesData.length;i++){
            console.log("i = ", architecturesData[i])
            GotArchitectures.push({name: architecturesData[i].name, architecture_data: architecturesData[i].architecture_data})
            let customOption = document.createElement("option");
            customOption.text = architecturesData[i].name
            customOption.value = architecturesData[i].name
            architectureNameCombo.appendChild(customOption)
        }
        console.log(GotArchitectures)

    }
    if (xhr.readyState == 4 && xhr.status == 400) {
        alert("wrong frontend get")
    }
    if (xhr.readyState == 4 && xhr.status == 500) {
        alert("wrong backend get")
    }
}

function getNumberOfArchitecturesFromBackend(){
    console.log("Getting data:");
    xhr = getXmlHttpRequestObject();
    xhr.onreadystatechange = GetDataCallback;
    // asynchronous requests
    xhr.open("GET", "http://localhost:6969/ArchitecturesNames", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    // Send the request over the network
    xhr.send();
}


downloadButton.onclick = getNumberOfArchitecturesFromBackend

function recreateArchitectureConnections(architecture){
    architecture=architecture[0]

    const all = document.getElementById("list").querySelectorAll(".layerType")

    for(let i=0;i<architecture.length;i++){
        let arch = architecture[i]
        if(!(architecture[i].constructor === Array)){

            if(i+1<architecture.length){
                let start
                let end
                for(let z=0;z<all.length;z++)
                {
                    console.log(all[z].id, arch.id)
                    if(all[z].id===arch.id){
                        start = all[z].querySelectorAll(".outputButton")[0]

                    }
                    if(all[z].id===architecture[i+1].id){
                        end = all[z].querySelectorAll(".inputButton")[0]
                    }
                }
                console.log(start, end)
                let recLine = reconstructStartLine(start)
                reconstructFinishLine(start, end, recLine)
                //reconstruct connection
            }
            else{
                //do nothing cause last element
            }
        }
        else{

        }
    }
}

function recreateArchitecture(architecture){
    architecture = architecture[0]
    ////hlip hlip hlip help me god
    console.log("HELP")
    console.log("")
    console.log(architecture)
    console.log(architecture[0])
    console.log("")
    let pageYjump = 0
    let pageXjump = 0
    for(let i=0;i<architecture.length;i++){

        if(!(architecture[i].constructor === Array)){
            reconstructMoreThanLayer(architecture[i],pageXjump +500, i*200 + 200 + pageYjump)
            // if((i+1)<architecture.length){
            //     if(!(architecture[i+1].constructor === Array)){

            //     }
            // }

        }
        else{
            pageYjump += test1recration(architecture[i], pageYjump, pageXjump, i)
            console.log(pageYjump, pageXjump)
        }

    }




}

function test1recration(architecture, pageYjump, pageXjump, shift){
    console.log(pageYjump,pageXjump)
    let size = 0
    for (let k = 0; k<architecture.length; k++){
        let arch = architecture[k]
        pageXjump += k*400
        for(let i=0;i<arch.length;i++){
            reconstructMoreThanLayer(arch[i], pageXjump + 500, i*200 + shift*200 + 200)
            if(size<i*200){
                size = i*200
            }
        }
    }
    return size

}

architectureNameCombo.addEventListener("change", recreateMoreThanArchitecture)

function recreateMoreThanArchitecture(){
    console.log("XDDDDDDDDDDDDD1")
    console.log(document.getElementById("list").querySelectorAll(".layerType"))
    console.log(elementsData)
    console.log(document.getElementById("list").querySelectorAll(".layerType").length)
    if(curentModel=="Default" && document.getElementById("list").querySelectorAll(".layerType").length>0){
        const index = GotArchitectures.findIndex(data => data.name === "Default");
        if (index !== -1) {
            GotArchitectures.splice(index, 1); // Remove the object from the array
        }
        let dataToSend = generateArchitectureJSON(elementsData, connections)
        const inputShapeInput = document.getElementsByClassName("ArchitectureShapeInput")[0]
        dataToSend[0][0].input_shape = inputShapeInput.value
        console.log(dataToSend)
        let dataToSend2 = {name: "Default", architecture_data: dataToSend}
        GotArchitectures.push(dataToSend2)
        console.log("GotArchitectures")
        console.log(GotArchitectures)
    }
    delLayersXD()
    for(let i=0; i<GotArchitectures.length; i++){
        if(GotArchitectures[i].name===architectureNameCombo.value){
            recreateArchitecture(GotArchitectures[i].architecture_data)
            recreateArchitectureConnections(GotArchitectures[i].architecture_data)
            curentModel = GotArchitectures[i].name

            break
        }
        curentModel="Default"
    }
    console.log("XDDDDDDDDDDDDD2")
    console.log(document.getElementById("list").querySelectorAll(".layerType"))
    console.log(elementsData)

}

function delLayersXD(){
    while(elementsData.length>0){
        console.log(elementsData.length)
        let elementId = elementsData[0].id
        const index = elementsData.findIndex(data => data.id === elementId);

        if (connections.length>0){
            delLine2(elementId)
        }

        if (index !== -1) {
            elementsData.splice(index, 1); // Remove the object from the array
        }
    }

    const all = document.getElementById("list").querySelectorAll(".layerType")
    for(let i=all.length-1; i>=0;i--){
        all[i].remove()
    }


    console.log(all)

}

function testCallback(){
    // Check response is ready or not
    if (xhr.readyState == 4 && xhr.status == 201) {
        console.log("create 201")
    }
    if (xhr.readyState == 4 && xhr.status == 400) {
        alert("create 400")
    }
    if (xhr.readyState == 4 && xhr.status == 500) {
        alert("create 500")
    }
}

function testPostMessage(){
    console.log("testPostMessage");
    xhr = getXmlHttpRequestObject();
    xhr.onreadystatechange = testCallback;
    xhr.open("POST", "http://localhost:6969/create", true);
    xhr.send();
}

testButton.onclick = testPostMessage