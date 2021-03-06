int i, j, k, k2, start_index, end_index, start_index2, end_index2, ith_step;
double grad_i_x, grad_i_y, align_i_x, align_i_y, f_i_x, f_i_y;
double beta_ij, ar_slope, ar_interc, r, temp, noise, c, s, v0_i;
double stat_align_x, stat_align_y, cm_x, cm_y, rel_pos_x, rel_pos_y;
double stat_angular, stat_clu, stat_radial_cf_sum, stat_pw_dist_sum;
double nearest_n, stat_nn_dist_sum, stat_radial_cm_sum;
int total_nb;

for (ith_step = 0; ith_step < steps; ith_step++) {
  stat_align_x = 0;
  stat_align_y = 0;
  stat_clu = 0;
  cm_x = 0;
  cm_y = 0;
  stat_radial_cf_sum = 0;
  stat_pw_dist_sum = 0.000000001;
  stat_nn_dist_sum = 0;

  // UPDATE DIRECTION
  start_index = 0;
  for (k = 0; k < 3; k++) {
    end_index = start_index + n_per_species[k];

    if (pinned[k] == 0) {
      // Only if i is not pinned

      // GRADIENT (constant across same species)
      grad_i_x = grad_x[k];
      grad_i_y = grad_y[k];

      for (i = start_index; i < end_index; i++) {
        // INITIALIZATION
        // Alignment term
        align_i_x = 0;
        align_i_y = 0;
        // A-R term
        f_i_x = 0;
        f_i_y = 0;
        // STAT_CLU
        total_nb = 0;
        // STAT_NN_DIST
        nearest_n = size_x + size_y;

        start_index2 = 0;
        for (k2 = 0; k2 < 3; k2++) {
          end_index2 = start_index2 + n_per_species[k2];

          beta_ij = beta[k*3 + k2];
          ar_slope = (1 + beta_ij) * f0 / (r1 - r0_x_2);
          ar_interc = - r0_x_2 * (1 + beta_ij) * f0 / (r1 - r0_x_2) - f0;

          for (j = start_index2; j < end_index2; j++) {
            if (i != j) {
              r = fb_dist(pos_x[i], pos_y[i], pos_x[j], pos_y[j]);
              // ALIGNMENT
              if (pinned[k2] == 0) {
                // Only if j is not pinned
                if (r <= rv) {
                  align_i_x += dir_x[j];
                  align_i_y += dir_y[j];
                }
                // STAT_PW_DIST
                stat_pw_dist_sum += r;
                // STAT_NN_DIST
                if (r < nearest_n) {
                  nearest_n = r;
                }
              }
              // ATTRACTION-REPULSION
              if (r <= r1) {
                if (r < r0_x_2) {
                  // Infinite repulsion
                  f_i_x += -10000 * (pos_x[j] - pos_x[i]);
                  f_i_y += -10000 * (pos_y[j] - pos_y[i]);
                } else {
                  // Equilibrium attraction and repulsion
                  if (r > 0) {
                    temp = r * ar_slope + ar_interc;
                    f_i_x += temp * (pos_x[j] - pos_x[i]) / r;
                    f_i_y += temp * (pos_y[j] - pos_y[i]) / r;
                  }
                }
                // STAT_CLU
                total_nb += 1;
              }
            }
          }
          start_index2 = end_index2;
        }

        // INERTIA
        dir_x[i] *= iner_coef;
        dir_y[i] *= iner_coef;

        // ADD OTHER TERMS
        dir_x[i] += grad_i_x + align_i_x*fa + f_i_x;
        dir_y[i] += grad_i_y + align_i_y*fa + f_i_y;

        // NORMALIZE (ARG)
        temp = sqrt(pow(dir_x[i], 2) + pow(dir_y[i], 2));
        if (temp > 0) {
          //Avoid dividing by zero
          dir_x[i] /= temp;
          dir_y[i] /= temp;
        }

        // NOISE
        noise = noise_coef*M_PI*((static_cast <float> (rand()) /
        static_cast <float> (RAND_MAX))*2-1);
        c = cos(noise);
        s = sin(noise);
        temp = dir_x[i];
        dir_x[i] = dir_x[i]*c - dir_y[i]*s;
        dir_y[i] = temp*s + dir_y[i]*c;

        // STAT_ALIGN
        stat_align_x += dir_x[i];
        stat_align_y += dir_y[i];

        // STAT_CLU
        stat_clu += total_nb;

        // STAT_NN_DIST
        stat_nn_dist_sum += nearest_n;
      }
    }

    start_index = end_index;
  }

  // UPDATE POSITION
  start_index = 0;
  for (k = 0; k < 3; k++) {
    end_index = start_index + n_per_species[k];

    if (pinned[k] == 0) {
      // Only if the cell type is not pinned

      // SPEED (constant across same species)
      v0_i = v0[k];

      for (i = start_index; i < end_index; i++) {
        pos_x[i] += v0_i * dir_x[i];
        pos_x[i] = fb_fitInto(pos_x[i], size_x);
        pos_y[i] += v0_i * dir_y[i];
        pos_y[i] = fb_fitInto(pos_y[i], size_y);

        //STAT_ANGULAR, STAT_CM
        cm_x += pos_x[i];
        cm_y += pos_y[i];

        //STAT_RADIAL_CF
        stat_radial_cf_sum += sqrt(pow((pos_x[i]-size_x/2),2)
          + pow((pos_y[i]-size_y/2),2));
      }
    }

    start_index = end_index;
  }

  // Update global stats
  if (eff_nop > 0) {
    // STAT_ANGULAR (0*steps+ith_step)
    cm_x /= eff_nop;
    cm_y /= eff_nop;
    stat_angular = 0;
    stat_radial_cm_sum = 0;
    for (i = 1; i < n; i++) {
      rel_pos_x = pos_x[i] - cm_x;
      rel_pos_y = pos_y[i] - cm_y;
      stat_angular += rel_pos_x * dir_y[i] - rel_pos_y * dir_x[i];
      // TODO: take into account pinned cells
      stat_radial_cm_sum += sqrt(pow(rel_pos_x,2) + pow((rel_pos_y),2));
    }
    global_stats[ith_step] = abs(stat_angular) / stat_radial_cm_sum;

    // STAT_ALIGN (1*steps+ith_step)
    global_stats[steps + ith_step] = sqrt(pow(stat_align_x, 2)
      + pow(stat_align_y, 2)) / eff_nop;

    // STAT_CLU (2*steps+ith_step)
    global_stats[2*steps + ith_step] = (stat_clu/
      (eff_nop*M_PI*r1*r1/(size_x*size_y)))/eff_nop;

    // STAT_CM (3, 4*steps+ith_step)
    global_stats[3*steps + ith_step] = cm_x;
    global_stats[4*steps + ith_step] = cm_y;

    // STAT_RADIAL_CF (5*steps+ith_step)
    global_stats[5*steps + ith_step] = stat_radial_cf_sum / eff_nop;

    // STAT_PW_DIST (6*steps+ith_step)
    global_stats[6*steps + ith_step] = stat_pw_dist_sum /
      (eff_nop*(eff_nop - 1));

    // STAT_NN_DIST (7*steps+ith_step)
    global_stats[7*steps + ith_step] = stat_nn_dist_sum / eff_nop;

    // STAT_MIGRATION (8*steps+ith_step)
    if (ith_step < 1) {
      global_stats[8*steps + ith_step] = 0;
    } else {
      global_stats[8*steps + ith_step] = fb_dist(cm_x, cm_y,
        global_stats[3*steps + ith_step-1],
        global_stats[4*steps + ith_step-1]);
    }

    // STAT_RADIAL_CM (9*steps+ith_step)
    global_stats[9*steps + ith_step] = stat_radial_cm_sum / eff_nop;
  }
}
