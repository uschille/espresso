

#include "external_potential.h"
#include "lattice.h"
#include "communication.h"
#include "integrate.h"

ExternalPotential* external_potentials;
int n_external_potentials;

void external_potential_pre_init() {
  printf("preinit external potential\n");
  external_potentials = NULL;
  n_external_potentials = 0;
}


int generate_external_potential(ExternalPotential** e) {
  external_potentials = realloc(external_potentials, (n_external_potentials+1)*sizeof(ExternalPotential));
  n_external_potentials++;


//  e = &external_potentials[n_external_potentials-1];
  return ES_OK;
}     

int external_potential_tabulated_init(int number, char* filename, int n_particle_types, double* scale) {

  ExternalPotentialTabulated* e = &external_potentials[number].e.tabulated;

  if (strlen(filename)>MAX_FILENAME_SIZE)
    return ES_ERROR;
  strcpy((char*)&(e->filename), filename);
  external_potentials[number].type=EXTERNAL_POTENTIAL_TYPE_TABULATED;
  external_potentials[number].scale = malloc(n_particle_types*sizeof(double));
  external_potentials[number].n_particle_types = n_particle_types;
  for (int i = 0; i < n_particle_types; i++) {
    external_potentials[number].scale[i]=scale[i];
  }
  mpi_external_potential_broadcast(number);
  mpi_external_potential_tabulated_read_potential_file(number);
  return ES_OK;
}


int external_potential_tabulated_read_potential_file(int number) {
  ExternalPotentialTabulated *e = &(external_potentials[number].e.tabulated);
//  printf("(%d) reading input file %s\n", this_node, e->filename);
  FILE* infile = fopen(e->filename, "r");
  
  if (!infile)  {
    char *errtxt = runtime_error(128+MAX_FILENAME_SIZE);
    ERROR_SPRINTF(errtxt,"Could not open file %s\n", e->filename);
    return ES_ERROR;
  }
  char first_line[100];
  char* token;
  double res[3];
  double size[3];
  double offset[3]={0,0,0};
  int dim=0;
  fgets(first_line, 100, infile);
  printf("(%d) reading %s\n", this_node, first_line);

  token = strtok(first_line, " \t");
  printf("(%d) %s\n", this_node, token);
  if (!token) { fprintf(stderr, "Error reading dimensionality\n"); return ES_ERROR; }
  dim = atoi(token);
  if (dim<=0)  { fprintf(stderr, "Error reading dimensionality\n"); return ES_ERROR; }
  
  token = strtok(NULL, " \t");
  printf("(%d) %s\n", this_node, token);
  if (!token) { fprintf(stderr, "Could not read box_l[0]\n"); return ES_ERROR; }
  size[0] = atof(token);

  token = strtok(NULL, " \t");
  printf("(%d) %s\n", this_node, token);
  if (!token) { fprintf(stderr, "Could not read box_l[1]\n"); return ES_ERROR; }
  size[1] = atof(token);
  
  token = strtok(NULL, " \t");
  printf("(%d) %s\n", this_node, token);
  if (!token) { fprintf(stderr, "Could not read box_l[2]\n"); return ES_ERROR;}
  size[2] = atof(token);

  token = strtok(NULL, " \t");
  printf("(%d) %s\n", this_node, token);
  if (!token) { fprintf(stderr, "Could not read res[0]\n"); return ES_ERROR;}
  res[0] = atof(token);
  
  token = strtok(NULL, " \t");
  printf("(%d) %s\n", this_node, token);
  if (!token) { fprintf(stderr, "Could not read res[1]\n"); return ES_ERROR;}
  res[1] = atof(token);
  
  token = strtok(NULL, " \t");
  printf("(%d) %s\n", this_node, token);
  if (!token) { fprintf(stderr, "Could not read res[2]\n"); return ES_ERROR;}
  res[2] = atof(token);

  token = strtok(NULL, " \t");
  if (token) {
    printf("Reading offset.\n");
    offset[0]=atof(token);
    token = strtok(NULL, " \t");
    if (!token) { fprintf(stderr, "Could not read offset[1]\n"); return ES_ERROR;}
    offset[1] = atof(token);
    token = strtok(NULL, " \t");
    if (!token) { fprintf(stderr, "Could not read offset[2]\n"); return ES_ERROR;}
    offset[2] = atof(token);
  }
  e->potential.offset[0]=offset[0];
  e->potential.offset[1]=offset[1];
  e->potential.offset[2]=offset[2];

  //printf("(%d) has read file %s\n", this_node, e->filename);

  //printf("Now it is time to initialize the lattice");
  int halosize=1;

  //printf("%f %f %f\n", size[0], size[1], size[2]);
  //printf("%f %f %f\n", box_l[0], box_l[1], box_l[2]);
  if (size[0] > 0 && abs(size[0] - box_l[0]) > ROUND_ERROR_PREC) {
    char *errtxt = runtime_error(128);
    ERROR_SPRINTF(errtxt,"Box size in x is wrong %f vs %f\n", size[0], box_l[0]);
    return ES_ERROR;
  }
  if (size[1] > 0 && abs(size[1] - box_l[1]) > ROUND_ERROR_PREC) {
    char *errtxt = runtime_error(128);
    ERROR_SPRINTF(errtxt,"Box size in y is wrong %f vs %f\n", size[1], box_l[1]);
    return ES_ERROR;
  }
  if (size[2] > 0 && abs(size[2] - box_l[2]) > ROUND_ERROR_PREC) {
    char *errtxt = runtime_error(128);
    ERROR_SPRINTF(errtxt,"Box size in z is wrong %f vs %f\n", size[2], box_l[2]);
    return ES_ERROR;
  }


  if (res[0] > 0)
    if (skin/res[0]>halosize) halosize = (int)ceil(skin/res[0]);
  if (res[1] > 0)
    if (skin/res[1]>halosize) halosize = (int)ceil(skin/res[1]);
  if (res[2] > 0)
    if (skin/res[2]>halosize) halosize = (int)ceil(skin/res[2]);

  printf("A good halo size is %d\n", halosize);

  // Now we count how many entries we have:

  init_lattice(&e->potential, res, offset, halosize, dim);
  e->potential.interpolation_type = INTERPOLATION_LINEAR;

  unsigned int linelength = (3+dim)*ES_DOUBLE_SPACE;
  char* line = malloc((3+dim)*ES_DOUBLE_SPACE);
  double pos[3];
  double f[3];
  int i;
  
  while (fgets(line, 200, infile)) {
//    printf("reading line %s\n", line);
    token = strtok(line, " \t");
    if (!token) { fprintf(stderr, "Could not read pos[0]\n"); return ES_ERROR; }
    pos[0] = atof(token);

    token = strtok(NULL, " \t");
    if (!token) { fprintf(stderr, "Could not read pos[1]\n"); return ES_ERROR; }
    pos[1] = atof(token);
    
    token = strtok(NULL, " \t");
    if (!token) { fprintf(stderr, "Could not read pos[1]\n"); return ES_ERROR; }
    pos[2] = atof(token);
    for (i=0; i<dim;i++) {
      token = strtok(NULL, " \t");
      if (!token) { fprintf(stderr, "Coud not read f[%d]\n", i); return ES_ERROR; }
      f[i] = atof(token);
    }
    lattice_set_data_for_global_position_with_periodic_image(&e->potential, pos, f);
  }
  free(line);

  check_lattice(&e->potential);
  
  if (check_runtime_errors()!=0)
    return ES_ERROR;
  return ES_OK;
}


int check_lattice(Lattice* lattice) {
  index_t index[3];
  double pos[3];
  int i,j,k;
  double *d;

  char filename[30];
  Lattice* l = lattice;
  sprintf(filename, "lattice_%02d.dat", this_node);
  FILE* outfile = fopen(filename , "w");
  fprintf(outfile,"grid %d %d %d\n", l->grid[0], l->grid[1], l->grid[2]); ;
  fprintf(outfile,"halo_grid %d %d %d\n", l->halo_grid[0], l->halo_grid[1], l->halo_grid[2]); ;
  fprintf(outfile,"halo_size %d\n", l->halo_size);
  
  fprintf(outfile,"grid_volume %ld\n", l->grid_volume);
  fprintf(outfile,"halo_grid_volume %ld\n", l->halo_grid_volume);
  fprintf(outfile,"halo_grid_surface %ld\n", l->halo_grid_surface);
  fprintf(outfile,"halo_offset %ld\n", l->halo_offset);

  fprintf(outfile,"dim %d\n", l->dim);

  fprintf(outfile,"agrid %f %f %f\n", l->agrid[0], l->agrid[1], l->agrid[2]);
 
  fprintf(outfile,"offset %f %f %f\n", l->offset[0], l->offset[1], l->offset[2]);
  fprintf(outfile,"local_offset %f %f %f\n", l->local_offset[0], l->local_offset[1], l->local_offset[2]);
  fprintf(outfile,"local_index_offset %d %d %d\n", l->local_index_offset[0], l->local_index_offset[1], l->local_index_offset[2]);


  fprintf(outfile, "element_size %d\n", l->element_size);
  fprintf(outfile, "lattice_dim %d\n", l->lattice_dim);

  
  for (i=0; i<lattice->halo_grid[0]; i++) 
    for (j=0; j<lattice->halo_grid[1]; j++) 
      for (k=0; k<lattice->halo_grid[2]; k++) {
        index[0]=i; index[1] = j; index[2] = k;
        lattice_get_data_for_halo_index(lattice, index, (void**) &d);
        map_halo_index_to_pos(lattice, index, pos);
//        map_local_index_to_pos(&e->lattice, index, pos);
        fprintf(outfile, "%f %f %f ||| %f \n",pos[0], pos[1], pos[2], d[0]);
      } 
  close(outfile);
  return ES_OK;
}

MDINLINE void add_external_potential_tabulated_forces(ExternalPotential* e, Particle* p) {
  printf("applying force\n");
  if (p->p.type >= e->n_particle_types) {
    return;
  }
  double field[3];
  double ppos[3];
  int img[3];
  memcpy(ppos, p->r.p, 3*sizeof(double));
  memcpy(img, p->r.p, 3*sizeof(int));
  fold_position(ppos, img);
 
  lattice_interpolate_gradient(&e->e.tabulated.potential, p->r.p, field);
  p->f.f[0]-=e->scale[p->p.type]*field[0];
  p->f.f[1]-=e->scale[p->p.type]*field[1];
  p->f.f[2]-=e->scale[p->p.type]*field[2];
}

void add_external_potential_forces(Particle* p) {
  for (int i=0; i<n_external_potentials; i++) {
    if (external_potentials[i].type==EXTERNAL_POTENTIAL_TYPE_TABULATED) {
      add_external_potential_tabulated_forces(&external_potentials[i], p);
    } else {
      char* c = runtime_error(128);
      ERROR_SPRINTF(c, "unknown external potential type");
    }
  }
}


MDINLINE void add_external_potential_tabulated_energy(ExternalPotential* e, Particle* p) {
  if (p->p.type >= e->n_particle_types) {
    return;
  }
  double potential;
  double ppos[3];
  int img[3];
  memcpy(ppos, p->r.p, 3*sizeof(double));
  memcpy(img, p->r.p, 3*sizeof(int));
  fold_position(ppos, img);
 
  lattice_interpolate(&e->e.tabulated.potential, p->r.p, &potential);
  e->energy += e->scale[p->p.type] * potential;
}

void add_external_potential_energy(Particle* p) {
  for (int i=0; i<n_external_potentials; i++) {
    if (external_potentials[i].type==EXTERNAL_POTENTIAL_TYPE_TABULATED) {
      add_external_potential_tabulated_energy(&external_potentials[i], p);
    } else {
      char* c = runtime_error(128);
      ERROR_SPRINTF(c, "unknown external potential type");
    }
  }
}

void external_potential_init_energies() {
  for (int i = 0; i<n_external_potentials; i++) {
    external_potentials[i].energy=0;
  }
}

