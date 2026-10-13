import os
import argparse
from string import Template
import subprocess
import getpass
import GPUtil

gpus = GPUtil.getGPUs()
gpu_ids = [gpu.id for gpu in gpus]
_user = getpass.getuser()
ARETOMO2 = f"/home/{_user}/Projects/AreTomo2/AreTomo2"

def align_tilt_series(st: str, xf: str, out: str, dry_run: bool = False):
    assert out.endswith(".mrc"), "Output file must have .mrc extension"
    _cmd = Template(
        "newstack -in $st -xf $xf -out $out"
    )
    _cmd = _cmd.substitute(st=st, xf=xf, out=out)
    print(f"{out}")
    print(f"Command: {_cmd}")
    if dry_run:
        return out
    subprocess.run(_cmd, shell=True)
    return out

def reconstruct_tomogram(
    input:str,
    tlt: str,
    out: str,
    px: float = 1.0,
    pertiltdose: float = 1.0,
    volZ: int = 200,
    binning: int = 4,
    ctf_estimation: bool = False,
    phase_estimation: bool = False,
    dry_run: bool = False
):
    assert out.endswith(".mrc"), "Output file must have .mrc extension"

    if ctf_estimation:

        _cmd = (
            f"{ARETOMO2} " +
            "-InMrc $input " +
            "-AngFile $tlt " +
            "-OutMrc $out " +
            "-FlipVol 1 " +
            "-volZ $volZ " +
            "-OutBin $binning " +
            f"-PixSize $px " +
            "-Wbp 1 " +
            f"-Gpu {",".join(str(id) for id in gpu_ids)} " +
            "-Align 1 " +
            "-Kv 300.0 " + # Only used for CTF estimation, not for alignment
            "-Cs 2.7 " + # Only used for CTF estimation, not for alignment
            "-AmpContrast 0.07 " + # Only used for CTF estimation, not for alignment
            "-ImgDose $pertiltdose " + # only used for dose weighting, not for alignment
            "-TiltAxis 0.0001 -1 "
        )

        if phase_estimation:
            _cmd += "-ExtPhase 10.0 180.0 " # for VPP

        _cmd = Template(_cmd)
        _cmd = _cmd.substitute(
            input=input,
            tlt=tlt,
            out=out,
            volZ=volZ,
            binning=binning,
            px=px,
            pertiltdose=pertiltdose
        )

    else:
        # needs modification in AreTomo2 to allow for no alignment
        _cmd = Template(
            f"{ARETOMO2} " +
            "-InMrc $input " +
            "-AngFile $tlt " +
            "-OutMrc $out " +
            "-FlipVol 1 " +
            "-volZ $volZ " +
            "-OutBin $binning " +
            "-Wbp 1 " +
            f"-Gpu {','.join(str(id) for id in gpu_ids)} " +
            "-Align 0 "
        )

        _cmd = _cmd.substitute(
            input=input,
            tlt=tlt,
            out=out,
            volZ=volZ,
            binning=binning
        )

    print(f"{out}")
    print(f"Command: {_cmd}")

    if dry_run:
        return out

    subprocess.run(_cmd, shell=True)
    return out

def get_preexposures(
    tlt: str,
    pertiltdose: float,
    scheme: str = "dose-symmetric"
):

    import numpy as np
    tlt_angles = np.loadtxt(tlt)
    if scheme == "dose-symmetric":
        pre_exposures = [0]*len(tlt_angles)
        # tilts start from 0 then go to positive and negative tilt angles
        # for e.g., 0, 2, -2, 4, -4, 6, -6, ...
        # we want dose exposures to be cumulative in this order of tilt
        sorted_indices = np.argsort(np.abs(tlt_angles))
        cumulative_dose = 0.0
        for idx in sorted_indices:
            pre_exposures[idx] = f"{float(cumulative_dose):.6f}"
            cumulative_dose += pertiltdose
        
    else:
        raise ValueError(f"Unsupported tilt scheme: {scheme}")

    return pre_exposures

def make_relion_star_files(
    tomo_path: str,
    px: float,
    tlt: str,
    ctf_txt: str,
    pertiltdose: float,
    micro_prefix: str,
    ctf_prefix: str,
    tilt_axis_angle: float = 0.0,
):

    import mrcfile
    with mrcfile.open(tomo_path, mode='r') as mrc:
        tomo_shape = mrc.data.shape
        dimz, dimx, dimy = tomo_shape

    tomodir = os.path.dirname(tomo_path)
    base_tomo = os.path.basename(tomo_path).replace('.mrc', '')
    base_tilt = os.path.basename(tomo_path).replace('.mrc', '').replace("tomo", "aligned")
    base = os.path.basename(tomo_path).replace('.mrc', '').replace("tomo", "")

    pre_exposures = get_preexposures(tlt, pertiltdose, "dose-symmetric")

    import numpy as np
    tlt_angles = np.loadtxt(tlt)
    ctf_params = np.loadtxt(ctf_txt)

    from cryoet_alignment.io.cryoet_data_portal import Alignment
    from cryoet_alignment import read, write
    _aln = Alignment.from_aretomo3(
        read(os.path.join(tomodir, f"{base_tilt}.aln")),
        vol_size=(dimx, dimy, dimz)
    )

    rel_aln = _aln.to_relion(
        tomo_name=base_tomo,
        pixel_size_a=px,
        hand=-1,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        pre_exposures=pre_exposures,
    )

    write(rel_aln, os.path.join(tomodir, "relion_out", f"{base_tomo}.star"))
    import starfile
    tiltstar = starfile.read(os.path.join(tomodir, "relion_out", "tilt_series", f"{base_tomo}.star"))

    tiltstar_fields = {
        # "rlnTomoNominalDefocus": [f"{float(expected_defocus):.6f}"] * len(tlt_angles),
        "rlnMicrographName": [
            f"{micro_prefix}/{base}{i:02d}.mrc"
            for i in range(1, len(tlt_angles)+1)
        ],
        "rlnTomoNominalTiltAxisAngle": [f"{float(tilt_axis_angle):.6f}"] * len(tlt_angles),
        "rlnCtfImage": [
            f"{int(i):08d}@{ctf_prefix}/{base_tilt}_ctf.mrc:mrcs"
            for i in range(1, len(tlt_angles)+1)
        ],
        "rlnDefocusU": ctf_params[:, 1],
        "rlnDefocusV": ctf_params[:, 2],
        "rlnDefocusAngle": ctf_params[:, 3],
        "rlnCtfFigureOfMerit": ctf_params[:, 5],
        "rlnCtfMaxResolution": ctf_params[:, 6],
        # "rlnCtfAstigmatism": [],
        # "rlnCtfIceRingDensity": [],
        # "rlnCtfScalefactor": [],
    }

    # add tiltstar_fields to tiltstar
    for key, value in tiltstar_fields.items():
        tiltstar[key] = value

    # write the updated tiltstar to a new file
    starfile.write(tiltstar, os.path.join(tomodir, "relion_out", "tilt_series", f"{base_tomo}.star"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-st",
        type=str,
        required=True,
        help="Path to the source tilt series (MRC stack) file.",
    )

    parser.add_argument(
        "-xf",
        type=str,
        required=True,
        help="Transformation file containing affine matrix",
    )

    parser.add_argument(
        "-tlt",
        type=str,
        required=True,
        help="Tilt angles file (text file with one angle per line).",
    )

    parser.add_argument(
        "-out",
        type=str,
        required=True,
        help="Output path for the reconstructed tomogram (MRC file).",
    )

    parser.add_argument(
        "-px",
        type=float,
        help="Pixel size of the input tilt series (in Angstroms).",
    )

    parser.add_argument(
        "-volZ",
        type=int,
        help="Z dimension of the output tomogram (in unbinned pixels).",
    )

    parser.add_argument(
        "-binning",
        type=int,
        default=4,
        help="Binning factor for the output tomogram.",
    )

    parser.add_argument(
        "-pertiltdose",
        type=float,
        default=1.0,
        help="Dose per tilt (in electrons per square Angstrom).",
    )

    parser.add_argument(
        "--ctf_estimation",
        action="store_true",
        help="Flag to enable CTF estimation during reconstruction.",
    )

    parser.add_argument(
        "--phase_estimation",
        action="store_true",
        help="Flag to enable phase estimation during CTF estimation.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, the script will only print the commands without executing them.",
    )

    args = parser.parse_args()

    outdir = os.path.dirname(args.out)
    os.makedirs(outdir, exist_ok=True)

    # Align the tilt series
    out_align = align_tilt_series(
        st=args.st,
        xf=args.xf,
        out=args.out.replace(".mrc", "_aligned.mrc"),
        dry_run=args.dry_run,
    )

    # Reconstruct the tomogram
    out_tomo = reconstruct_tomogram(
        input=out_align,
        tlt=args.tlt,
        out=args.out.replace(".mrc", "_tomo.mrc"),
        px=args.px,
        pertiltdose=args.pertiltdose,
        volZ=args.volZ,
        binning=args.binning,
        ctf_estimation=args.ctf_estimation,
        phase_estimation=args.phase_estimation,
        dry_run=args.dry_run,
    )

    make_relion_star_files(
        tomo_path=out_tomo,
        px=args.px,
        tlt=args.tlt,
        ctf_txt=args.out.replace(".mrc", "_aligned_ctf.txt"),
        pertiltdose=args.pertiltdose,
        micro_prefix="frames",
        ctf_prefix="CTF",
    )