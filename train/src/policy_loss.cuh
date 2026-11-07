extern "C" __global__ void policy_loss(const float *z, const float *targets,
                                       const int *ranges, const int *moves,
                                       float *out) {
  int b = blockIdx.x * THREADS + threadIdx.x;
  if (b >= BATCH)
    return;
  int start = ranges[2 * b];
  int count = ranges[2 * b + 1];
  if (count == 0) {
    out[b] = 0;
    return;
  }
  z += (size_t)b * MOVES;
  moves += start;
  targets += start;
  float max = z[moves[0]], sum = 0;
  for (int s = 1; s < count; s++)
    max = fmaxf(max, z[moves[s]]);
  for (int s = 0; s < count; s++)
    sum += expf(z[moves[s]] - max);

  float cross = 0;
  float entropy = 0;
  for (int s = 0; s < count; s++) {
    cross += targets[s] * (z[moves[s]] - max);
    if (targets[s] > 0)
      entropy -= targets[s] * logf(targets[s]);
  }
  out[b] = logf(sum) - cross - entropy;
}

extern "C" __global__ void policy_loss_bwd(const float *z, const float *targets,
                                           const int *ranges, const int *moves,
                                           const float *grad, float *dz) {
  int b = blockIdx.x * THREADS + threadIdx.x;
  if (b >= BATCH)
    return;
  int start = ranges[2 * b];
  int count = ranges[2 * b + 1];
  if (count == 0)
    return;
  z += (size_t)b * MOVES;
  dz += (size_t)b * MOVES;
  moves += start;
  targets += start;
  float max = z[moves[0]], sum = 0;
  for (int s = 1; s < count; s++)
    max = fmaxf(max, z[moves[s]]);
  for (int s = 0; s < count; s++)
    sum += expf(z[moves[s]] - max);

  for (int s = 0; s < count; s++) {
    dz[moves[s]] = grad[0] * (expf(z[moves[s]] - max) / sum - targets[s]);
  }
}
