// Display-only reslicing of the downsampled viewer MRC. Scientific maps are untouched.
export function parseViewer(buffer) {
  const header = new DataView(buffer);
  const nx = header.getInt32(0, true), ny = header.getInt32(4, true), nz = header.getInt32(8, true);
  const offset = 1024 + header.getInt32(92, true);
  if (header.getInt32(12, true) !== 2 || offset + nx*ny*nz*4 > buffer.byteLength) throw new Error('Invalid viewer map');
  const data = new Float32Array(buffer, offset, nx*ny*nz);
  const voxel = header.getFloat32(40, true) / header.getInt32(28, true);
  let sum = 0, sum2 = 0;
  for (const v of data) { sum += v; sum2 += v*v; }
  const mean = sum/data.length, sigma = Math.sqrt(Math.max(0, sum2/data.length-mean*mean));
  return {data, nx, ny, nz, voxel, mean, sigma};
}
function sample(v, x, y, z) {
  if (x<0 || y<0 || z<0 || x>v.nx-1 || y>v.ny-1 || z>v.nz-1) return 0;
  const x0=Math.floor(x), y0=Math.floor(y), z0=Math.floor(z);
  const dx=x-x0, dy=y-y0, dz=z-z0;
  let sum=0;
  for (let k=0;k<2;k++) for(let j=0;j<2;j++) for(let i=0;i<2;i++) {
    const ix=Math.min(x0+i,v.nx-1), iy=Math.min(y0+j,v.ny-1), iz=Math.min(z0+k,v.nz-1);
    sum += v.data[(iz*v.ny+iy)*v.nx+ix] * (i?dx:1-dx) * (j?dy:1-dy) * (k?dz:1-dz);
  }
  return sum;
}
export function drawSlice(canvas, volume, rotation, depth) {
  const {nx,ny,nz} = volume;
  const r = rotation ? [...rotation] : [1,0,0,0,1,0,0,0,1];
  for(let j=0;j<3;j++) {
    const norm=Math.hypot(r[j], r[3+j], r[6+j]) || 1;
    for(let i=0;i<3;i++) r[i*3+j] /= norm;
  }
  // The negated screen-right axis preserves the reference renderer's handedness.
  const extent=Math.max(nx,ny,nz)*0.5*1.05;
  const half=Math.min(Math.round(extent),128), size=half*2+1;
  const off=(depth/100-0.5)*2*extent;
  const coords=(col,row)=>{
    const a=(col-half)*extent/half, b=(half-row)*extent/half;
    return [(nx-1)/2+off*r[6]-a*r[0]+b*r[3],
            (ny-1)/2+off*r[7]-a*r[1]+b*r[4],
            (nz-1)/2+off*r[8]-a*r[2]+b*r[5]];
  };
  const values=new Float32Array(size*size);
  for(let y=0;y<size;y++) for(let x=0;x<size;x++) values[y*size+x]=sample(volume,...coords(x,y));
  const sorted=Float32Array.from(values).sort();
  const percentile=p=>{const q=(sorted.length-1)*p, i=Math.floor(q); return sorted[i]+(sorted[Math.min(i+1,sorted.length-1)]-sorted[i])*(q-i);};
  const low=percentile(.01), high=percentile(.995), range=high-low;
  canvas.width=canvas.height=size;
  const ctx=canvas.getContext('2d'), image=ctx.createImageData(size,size);
  for(let i=0;i<values.length;i++) {
    const g=range>0?Math.round(255*Math.max(0,Math.min(1,(values[i]-low)/range))):0;
    image.data.set([g,g,g,255],i*4);
  }
  ctx.putImageData(image,0,0);
  canvas.sliceInfo = {volume, values, size, coords, depth, offset:off};
}
