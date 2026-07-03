/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as d3 from 'd3';

/**
 * A two dimensional example: x and y coordinates with the label.
 */
export type Example2D = {
  x: number,
  y: number,
  label: number
};

type Point = {
  x: number,
  y: number
};

/**
 * Shuffles the array using Fisher-Yates algorithm. Uses the seedrandom
 * library as the random generator.
 */
export function shuffle(array: any[]): void {
  let counter = array.length;
  let temp = 0;
  let index = 0;
  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = Math.floor(Math.random() * counter);
    // Decrease counter by 1
    counter--;
    // And swap the last element with it
    temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

export type DataGenerator = (numSamples: number, noise: number) => Example2D[];

export function classifyTwoGaussData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];

  let varianceScale = d3.scale.linear().domain([0, .5]).range([0.5, 4]);
  let variance = varianceScale(noise);

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      let x = normalRandom(cx, variance);
      let y = normalRandom(cy, variance);
      points.push({x, y, label});
    }
  }

  genGauss(2, 2, 1); // Gaussian with positive examples.
  genGauss(-2, -2, -1); // Gaussian with negative examples.
  return points;
}

export function classifyTwoGaussDataTwo(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];

  let varianceScale = d3.scale.linear().domain([0, .5]).range([0.5, 4]);
  let variance = varianceScale(noise);

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      let x = normalRandom(cx, variance);
      let y = normalRandom(cy, variance);
      points.push({x, y, label});
    }
  }

  genGauss(0, 0, -1); // Gaussian with negative examples.
  genGauss(-2, -2, 1); // Gaussian with positive examples.
  return points;
}

export function regressTwoClassData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];

  let varianceScale = d3.scale.linear().domain([0, .5]).range([0.5, 4]);
  let variance = varianceScale(noise);

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      let x = normalRandom(cx, variance);
      let y = normalRandom(cy, variance);
      points.push({x, y, label});
    }
  }

  genGauss(2, 2, 1); // Gaussian with positive examples.
  genGauss(-2, -2, -1); // Gaussian with negative examples.
  return points;
}

export function regressTwoClassDataTwo(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];

  let varianceScale = d3.scale.linear().domain([0, .5]).range([0.5, 4]);
  let variance = varianceScale(noise);

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      let x = normalRandom(cx, variance);
      let y = normalRandom(cy, variance);
      points.push({x, y, label});
    }
  }

  genGauss(0, 0, -1); // Gaussian with negative examples.
  genGauss(-2, -2, 1); // Gaussian with positive examples.
  return points;
}

/**
 * Fixed teaching dataset "Car": car price vs. mileage.
 * Eight raw (mileage, price) points, rescaled into the [-6, 6] plot window.
 * numSamples and noise are ignored; the points are listed twice so that both
 * the train and test split get full coverage.
 */
export function regressCarData(numSamples: number, noise: number):
  Example2D[] {
  let points: Example2D[] = [];
  let x_array = [25000, 34000, 45000, 70000, 93000, 110000, 125000, 160000,
    25000, 34000, 45000, 70000, 93000, 110000, 125000, 160000];
  let label_array = [23000, 28000, 22000, 22000, 12500, 5500, 5000, 4000,
    23000, 28000, 22000, 22000, 12500, 5500, 5000, 4000];
  numSamples = label_array.length;
  for (let i = 0; i < numSamples; i++) {
    let x = x_array[i]*6/100000 - 5;
    let y = 0;
    let label = label_array[i]*2/10000;
    points.push({x, y, label});
  }
  return points;
}

/**
 * Fixed teaching dataset "Three neuron": a RELU-shaped target (from
 * regression8.csv) that a small network can fit with a few neurons.
 * The raw points span x in [-9.5, 9.9] and label in [-16.8, 4.4], so each axis
 * is rescaled (keeping 0 -> 0) to fit inside the [-6, 6] plot window.
 * numSamples and noise are ignored; the points are listed twice so that both
 * the train and test split get full coverage.
 */
export function regressThreeNeuron(numSamples: number, noise: number):
  Example2D[] {
  // Raw (x, label) points from regression8.csv.
  let rawPoints: [number, number][] = [
    [-8, 0], [4.2, -5.4], [-7.7, 0], [-6.1, 0], [8.7, -14.4],
    [3.4, -3.8], [9, -15], [-5, 0], [-6.1, 0], [8.3, -13.6],
    [9.9, -16.8], [-2.3, 1.4], [-4.4, 0], [-1.4, 3.2], [1, 1],
    [-4.8, 0], [-6.5, 0], [-9.5, 0], [-6.3, 0], [4.6, -6.2],
    [5.5, -8], [-6.2, 0], [-0.8, 4.4], [-8.3, 0], [4.1, -5.2],
    [3.2, -3.4], [1.8, -0.6], [-9.2, 0], [-2.9, 0.2], [2.7, -2.4]
  ];
  // Scale factors that map the raw ranges into the [-6, 6] plot window.
  let xScale = 0.55;
  let labelScale = 0.32;
  let points: Example2D[] = [];
  // List the points twice so both the train and test split get full coverage.
  for (let pass = 0; pass < 2; pass++) {
    rawPoints.forEach(([rawX, rawLabel]) => {
      let x = rawX * xScale;
      let y = 0;
      let label = rawLabel * labelScale;
      points.push({x, y, label});
    });
  }
  return points;
}


export function regressLine(numSamples: number, noiseLevel: number):
  Example2D[] {
  let radius = 6;
  let labelScale = d3.scale.linear()
    .domain([-10, 10])
    .range([-10, 10]);
  let getLabel = (x) => labelScale(-0.5*x+1);

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = 0;
    // let noise = randUniform(-radius, radius) * noise;
    let noise = normalRandom(0, 1) * noiseLevel;
    let label = getLabel(x)+noise;
    points.push({x, y, label});
  }
  return points;
}

export function regressQuadratic(numSamples: number, noiseLevel: number):
  Example2D[] {
  let radius = 6;
  let labelScale = d3.scale.linear()
    .domain([-10, 10])
    .range([-10, 10]);
  let getLabel = (x) => labelScale(0.5*x*x-3);

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = 0;
    // let noise = randUniform(-radius, radius) * noise;
    let noise = normalRandom(0, 1) * noiseLevel;
    let label = getLabel(x)+noise;
    points.push({x, y, label});
  }
  return points;
}

export function regressSawtooth(numSamples: number, noiseLevel: number):
  Example2D[] {
  let points: Example2D[] = [];

  let labelScale = d3.scale.linear()
    .domain([0, 2])
    .range([5, 0])
    .clamp(true);

  let gaussians = [
    [-4, 1],
    [0, -1],
    [4, 1],
    [-4, -1],
    [0, 1],
    [4, -1]
  ];

  function getLabel(x: number) {
    // Choose the one that is maximum in abs value.
    let label = 0;
    gaussians.forEach(([cx, sign]) => {
      let newLabel = sign * labelScale(Math.abs(x-cx));
      if (Math.abs(newLabel) > Math.abs(label)) {
        label = newLabel;
      }
    });
    return label;
  }
  let radius = 6;
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-radius, radius);
    let y = 0;
    // let noiseX = randUniform(-radius, radius) * noiseLevel;
    let noise = normalRandom(0, 1) * noiseLevel;
    let label = getLabel(x) + noise;
    points.push({x, y, label});
  };
  return points;
}

export function classifySpiralData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];
  let n = numSamples / 2;

  function genSpiral(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      let r = i / n * 5;
      let t = 1.75 * i / n * 2 * Math.PI + deltaT;
      let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      let y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push({x, y, label});
    }
  }

  genSpiral(0, 1); // Positive examples.
  genSpiral(Math.PI, -1); // Negative examples.
  return points;
}

export function classifyCircleData(numSamples: number, noise: number):
    Example2D[] {
  let points: Example2D[] = [];
  let radius = 5;
  function getCircleLabel(p: Point, center: Point) {
    return (dist(p, center) < (radius * 0.5)) ? 1 : -1;
  }

  // Generate positive points inside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(0, radius * 0.5);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }

  // Generate negative points outside the circle.
  for (let i = 0; i < numSamples / 2; i++) {
    let r = randUniform(radius * 0.7, radius);
    let angle = randUniform(0, 2 * Math.PI);
    let x = r * Math.sin(angle);
    let y = r * Math.cos(angle);
    let noiseX = randUniform(-radius, radius) * noise;
    let noiseY = randUniform(-radius, radius) * noise;
    let label = getCircleLabel({x: x + noiseX, y: y + noiseY}, {x: 0, y: 0});
    points.push({x, y, label});
  }
  return points;
}

export function classifyXORData(numSamples: number, noise: number):
    Example2D[] {
  function getXORLabel(p: Point) { return p.x * p.y >= 0 ? 1 : -1; }

  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let x = randUniform(-5, 5);
    let padding = 0.3;
    x += x > 0 ? padding : -padding;  // Padding.
    let y = randUniform(-5, 5);
    y += y > 0 ? padding : -padding;
    let noiseX = randUniform(-5, 5) * noise;
    let noiseY = randUniform(-5, 5) * noise;
    let label = getXORLabel({x: x + noiseX, y: y + noiseY});
    points.push({x, y, label});
  }
  return points;
}

/**
 * Returns a sample from a uniform [a, b] distribution.
 * Uses the seedrandom library as the random generator.
 */
function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}

/**
 * Samples from a normal distribution. Uses the seedrandom library as the
 * random generator.
 *
 * @param mean The mean. Default is 0.
 * @param variance The variance. Default is 1.
 */
function normalRandom(mean = 0, variance = 1): number {
  let v1: number, v2: number, s: number;
  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1);

  let result = Math.sqrt(-2 * Math.log(s) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}

/** Returns the eucledian distance between two points in space. */
function dist(a: Point, b: Point): number {
  let dx = a.x - b.x;
  let dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}
