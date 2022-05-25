let draw_canvas, show_canvas;
let draw_ctx, show_ctx;
let h, w;
let flag = false, prevX = 0, currX = 0, prevY = 0, currY = 0, dot_flag = false;

let strokes = [];
let mode = "generate";
let category = "cat";
let model = "trans";

const strokeColor = "black";
const strokeWidth = 2;

function init() {
    draw_canvas = document.getElementById('draw_canvas');
    draw_ctx = draw_canvas.getContext("2d");

    show_canvas = document.getElementById('show_canvas');
    show_ctx = show_canvas.getContext("2d");

    w = draw_canvas.width;
    h = draw_canvas.height;

    draw_canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    draw_canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    draw_canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    draw_canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);
}


function draw() {
    draw_ctx.beginPath();
    draw_ctx.moveTo(prevX, prevY);
    draw_ctx.lineTo(currX, currY);
    draw_ctx.strokeStyle = strokeColor;
    draw_ctx.lineWidth = strokeWidth;
    draw_ctx.stroke();
    draw_ctx.closePath();
}

function select_mode() {
    const e = document.getElementById("mode");
    mode = e.options[e.selectedIndex].text;
}

function select_category() {
    const e = document.getElementById("category");
    category = e.options[e.selectedIndex].text;
}

function select_model() {
    const e = document.getElementById("model");
    model = e.options[e.selectedIndex].text;
}

function erase() {
    draw_ctx.clearRect(0, 0, w, h);
    show_ctx.clearRect(0, 0, w, h);
    strokes = [];
    flag = false;
    prevX = 0;
    currX = 0;
    prevY = 0;
    currY = 0;
    dot_flag = false;
}

async function fetch_data() {
    let data = {"strokes": (strokes.length === 0) ? null : strokes, "type": model, "version": "latest"}
    let response = await fetch(`http://127.0.0.1:8000/${mode}/${category}`, {
        method: "POST", headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
    })
    return await response.json()
}

function draw_all_strokes(data, startX, startY, factor) {
    show_ctx.beginPath();
    show_ctx.moveTo(startX, startY);
    show_ctx.strokeStyle = strokeColor;
    show_ctx.lineWidth = strokeWidth;
    let X = startX;
    let Y = startY;
    let lift_pen = 1;
    let command = "m";
    for (const triplet of data) {
        if (lift_pen === 1) {
            command = "m";
        } else if (command !== "l") {
            command = "l";
        } else {
            command = "l";
        }

        X += triplet[0] * factor;
        Y += triplet[1] * factor;
        lift_pen = triplet[2];

        if (command === 'm') {
            show_ctx.moveTo(X, Y);
        } else {
            show_ctx.lineTo(X, Y);
        }
    }
    show_ctx.stroke();
    show_ctx.closePath();
}

async function run() {
    show_ctx.clearRect(0, 0, w, h);
    let response = await fetch_data();
    const [minX, maxX, minY, maxY] = response['bounds']
    let factor = Math.min((h / 2) / (maxY - minY), (w / 2) / (maxX - minX));
    draw_all_strokes(response['strokes'], w / 4 - minX * factor, h / 4 - minY * factor, factor);
}

function findxy(res, e) {
    if (res === 'down') {
        prevX = currX;
        prevY = currY;
        currX = e.clientX - draw_canvas.offsetLeft;
        currY = e.clientY - draw_canvas.offsetTop;

        flag = true;
        dot_flag = true;
        if (dot_flag) {
            draw_ctx.beginPath();
            draw_ctx.fillStyle = strokeColor;
            draw_ctx.fillRect(currX, currY, 2, 2);
            draw_ctx.closePath();
            dot_flag = false;
        }
        if (strokes.length > 0) {
            strokes[strokes.length - 1][2] = 1;
        }
        strokes.push([currX - prevX, currY - prevY, 0]);
    }
    if (res === 'up' || res === "out") {
        flag = false;
    }
    if (res === 'move') {
        if (flag) {
            prevX = currX;
            prevY = currY;
            currX = e.clientX - draw_canvas.offsetLeft;
            currY = e.clientY - draw_canvas.offsetTop;
            draw();
            strokes.push([currX - prevX, currY - prevY, 0]);
        }
    }
}