export function prediksi(model, nilai) {
  if (nilai.length !== model.mean.length ||
      nilai.some(v => !Number.isFinite(v) || v < 0.01 || v > 20)) {
    throw new Error("Isi empat pengukuran dalam rentang 0,01–20 cm.");
  }
  const z = nilai.map((v, j) => (v - model.mean[j]) / model.scale[j]);
  const jarak = model.vectors.map((x, i) => ({
    i, d: x.reduce((s, v, j) => s + (v - z[j]) ** 2, 0)
  }));
  jarak.sort((a, b) => a.d - b.d || a.i - b.i);
  const suara = model.classes.map(() => 0);
  for (const {i} of jarak.slice(0, model.k)) suara[model.labels[i]]++;
  return suara.indexOf(Math.max(...suara));
}
