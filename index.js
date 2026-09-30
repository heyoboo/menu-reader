// index.mjs

import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  TextStreamer,
  load_image,
} from "@huggingface/transformers";
import { readFile } from "fs/promises";

// import pkg from "wavefile";
// const { WaveFile } = pkg;

// import { readFileSync, existsSync, createWriteStream } from "fs";
// import { pipeline } from "stream/promises";
// import { tmpdir } from "os";
// import { join } from "path";


// ANSI console formatting
const RESET  = "\x1b[0m";
const BOLD   = "\x1b[1m";
const GREEN  = "\x1b[32m";
const CYAN   = "\x1b[36m";
const YELLOW = "\x1b[33m";
const DIM    = "\x1b[2m";


// Logging helpers
function log(label, msg = "") {
  console.log(`${BOLD}${CYAN}[${label}]${RESET} ${msg}`);
}

function success(label, msg = "") {
  console.log(`${BOLD}${GREEN}[${label}]${RESET} ${msg}`);
}

function warn(msg) {
  console.log(`${BOLD}${YELLOW}[WARN]${RESET} ${msg}`);
}


// Execution-time helper
function elapsed(start) {
  return `${DIM}(${((Date.now() - start) / 1000).toFixed(1)}s)${RESET}`;
}


// Download progress
let progressStarted = false;

function renderOverallProgress(pct) {
  const clamped = Math.min(Math.round(pct ?? 0), 100);
  const filled = Math.round(clamped / 2);
  const bar = "█".repeat(filled) + "░".repeat(50 - filled);

  if (!progressStarted) {
    process.stdout.write("\n");
    progressStarted = true;
  }

  process.stdout.write(
    `\r  Downloading  [${CYAN}${bar}${RESET}] ${clamped
      .toString()
      .padStart(3)}%`
  );

  if (clamped >= 100) {
    process.stdout.write(`  ${GREEN}✓${RESET}\n`);
  }
}


// Load WAV audio manually for Node.js
// async function loadAudioNode(url) {
//   const tmpPath = join(tmpdir(), "audio_input.wav");

//   const res = await fetch(url);

//   if (!res.ok) {
//     throw new Error(`Failed to fetch audio: ${res.statusText}`);
//   }

//   const dest = createWriteStream(tmpPath);

//   await pipeline(res.body, dest);

//   const wav = new WaveFile(readFileSync(tmpPath));

//   wav.toBitDepth("32f");
//   wav.toSampleRate(16000);

//   let samples = wav.getSamples();

//   // If stereo, use only the first channel
//   if (Array.isArray(samples)) {
//     samples = samples[0];
//   }

//   return new Float32Array(samples);
// }


// Model
const model_id = "onnx-community/gemma-4-E2B-it-ONNX";


// 1. Processor
log(
  "1/5",
  `Loading processor from ${BOLD}${model_id}${RESET} ...`
);

let t = Date.now();

const processor = await AutoProcessor.from_pretrained(model_id);

success("1/5", `Processor ready ${elapsed(t)}`);


// 2. Model
log(
  "2/5",
  `Loading model (dtype=q4f16, device=cpu) — this may take several minutes on first run ...`
);

warn(
  "The ONNX model files are large. Progress bar will appear below as files download/load."
);

t = Date.now();

const model = await Gemma4ForConditionalGeneration.from_pretrained(
  model_id,
  {
    dtype: "q4f16",
    device: "cpu",

    progress_callback: (info) => {
      if (info.status === "progress_total") {
        renderOverallProgress(info.progress);
      }

      // suppress all other per-file noise
    },
  }
);

success("2/5", `Model loaded ${elapsed(t)}`);


// 3. Prompt
log("3/5", "Building chat prompt ...");

const messages = [
  {
    role: "user",
    content: [
      { type: "image" },
    //   { type: "audio" },
      {
        type: "text",
        // text: "Describe this image in detail and transcribe this audio verbatim.",
        text: "Describe this image in detail, and calculate the total price",
      },
    ],
  },
];

const prompt = processor.apply_chat_template(messages, {
  enable_thinking: false,
  add_generation_prompt: true,
});

success("3/5", "Prompt ready");


// 4. Inputs
log(
  "4/5",
  //   "Fetching image and audio, then processing inputs ..."
  "Fetching image, then processing inputs ..."
);

t = Date.now();

const imagePath = "./images/order_sheet1.png";

// const imageUrl =
//   "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/artemis.jpeg";

// const audioUrl =
//   "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav";


// Load image
log("4/5", `  Image → ${imagePath}`);

const image = await load_image(imagePath);

success("4/5", "  Image loaded");


// Load audio
// log("4/5", `  Audio → ${audioUrl}`);

// const audio = await loadAudioNode(audioUrl);

// success(
//   "4/5",
//   `  Audio loaded — ${audio.length} samples @ 16kHz`
// );


// Process multimodal inputs
const inputs = await processor(
  prompt,
  image,
//   audio,
  undefined,
  {
    add_special_tokens: false,
  }
);

success("4/5", `Inputs ready ${elapsed(t)}`);


// 5. Generate
log(
  "5/5",
  "Running inference (CPU can be slow — please wait) ..."
);

warn("Token generation will stream below as it completes:\n");

t = Date.now();

let tokenCount = 0;

const outputs = await model.generate({
  ...inputs,

  max_new_tokens: 512,

  do_sample: false,

  streamer: new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: false,

    callback_function: (text) => {
      tokenCount++;
      process.stdout.write(text);
    },
  }),
});

process.stdout.write("\n\n");

success(
  "5/5",
  `Generation done — ${tokenCount} tokens ${elapsed(t)}`
);


// Final decoded output
const decoded = processor.batch_decode(
  outputs.slice(
    null,
    [inputs.input_ids.dims.at(-1), null]
  ),
  {
    skip_special_tokens: true,
  }
);

console.log(
  `\n${BOLD}── Final Output ──────────────────────────────────${RESET}`
);

console.log(decoded[0]);

console.log(
  `${BOLD}──────────────────────────────────────────────────${RESET}\n`
);