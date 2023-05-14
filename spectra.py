import pathlib
import os
import numpy
from scipy import optimize
from matplotlib import pyplot as plt


def make_model(image, model, init_parameters):
    x = numpy.linspace(0, len(image), len(image))
    popt, pcov = optimize.curve_fit(model, x, image, init_parameters)
    data_fitted = model(x, *popt)
    return popt, data_fitted


def make_gaussian(x, x0, std, max_intensity):
    return max_intensity * numpy.exp(-(x - x0) ** 2 / 2 / std ** 2)


file_names = [str(_) for _ in pathlib.Path("../Data").glob("*.csv")]
spectra = []
for f in file_names:
    data = []
    x_axis = []
    with open(f, "r") as inp:
        print(f"Reading {f}...")
        for s in inp:
            if any(map(str.isdigit, s)):
                ss = s.split(",")
                try:
                    x_axis.append(float(ss[0].replace('"', " ").strip()))
                    data.append(float(ss[1].replace('"', " ").strip()))
                except ValueError:
                    print(f"Error! Unexpected value in csv file: {s}")
    spectra.append([x_axis, data])
h_alpha = 6563
h_beta = 4861
h_gamma = 4341
ca_v = 6087
ar_v_1 = 7005
ar_v_2 = 6434
k_iv = 6101
cl_iv_1 = 7531
cl_iv_2 = 8046
he_ii = 4686
s_ii_1 = 6716
s_ii_2 = 6731
for i in range(len(spectra)):
    print(f"Building spectrum: {os.path.basename(file_names[i]).split('.')[0]}...")
    plt.plot(spectra[i][0], spectra[i][1], linewidth=1)
    plt.title(os.path.basename(file_names[i]).split(".")[0])
    plt.xlabel("λ, angstrom")
    plt.ylabel("intensity")
    fig = plt.gcf()
    fig.set_size_inches(24, 12)
    fig.savefig(f"../spectra/{os.path.basename(file_names[i]).split('.')[0]}.png", dpi=500)
    fig.clear()
    with open(f"../spectra/{os.path.basename(file_names[i]).split('.')[0]}.dat", "w") as out:
        d_lambda = (spectra[i][0][-1] - spectra[i][0][0]) / (len(spectra[i][0]) - 1)
        print(f"d_lambda = {d_lambda}\n")
        out.write(f"d_lambda = {d_lambda}\n\n")
        mean = numpy.median(spectra[i][1])
        sigma = numpy.std(spectra[i][1])
        sigma = numpy.std(numpy.array(spectra[i][1])[numpy.array(spectra[i][1]) < mean + sigma])
        mean = numpy.median(numpy.array(spectra[i][1])[numpy.array(spectra[i][1]) < mean + sigma])
        z = []
        found_h_alpha = False
        found_h_beta = False
        found_h_gamma = False
        found_s_ii_1 = False
        found_s_ii_2 = False
    
        center = round((h_alpha - spectra[i][0][0]) / d_lambda)
        line_img = numpy.array(spectra[i][1][(center - 5):(center + 5)])
        try:
            params, gaussian_model = make_model(line_img, make_gaussian, [5, 1, spectra[i][1][5]])
            h_alpha_pr = (center - 5 + params[0]) * d_lambda + spectra[i][0][0]
            h_alpha_int = params[2]
            plt.plot(spectra[i][0][(center - 5):(center + 5)], line_img)
            plt.plot(spectra[i][0][(center - 5):(center + 5)], gaussian_model, color="#FF0000")
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.savefig(f"../spectra/tmp/{os.path.basename(file_names[i]).split('.')[0]}_h_alpha.png", dpi=200)
            fig.clear()
            if h_alpha_int > mean + sigma:
                print(f"H alpha: {h_alpha_pr} (z = {(h_alpha_pr - h_alpha) / h_alpha})")
                out.write(f"H alpha: {h_alpha_pr} (z = {(h_alpha_pr - h_alpha) / h_alpha})\n")
                z.append((h_alpha_pr - h_alpha) / h_alpha)
                found_h_alpha = True
            else:
                print("H alpha: too low intensity!")
                out.write("H alpha: too low intensity!\n")
        except RuntimeError:
            print("H alpha: ???")
            out.write("H alpha: ???\n")
    
        center = round((h_beta - spectra[i][0][0]) / d_lambda)
        line_img = numpy.array(spectra[i][1][(center - 5):(center + 5)])
        try:
            params, gaussian_model = make_model(line_img, make_gaussian, [5, 1, spectra[i][1][5]])
            h_beta_pr = (center - 5 + params[0]) * d_lambda + spectra[i][0][0]
            h_beta_int = params[2]
            plt.plot(spectra[i][0][(center - 5):(center + 5)], line_img)
            plt.plot(spectra[i][0][(center - 5):(center + 5)], gaussian_model, color="#FF0000")
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.savefig(f"../spectra/tmp/{os.path.basename(file_names[i]).split('.')[0]}_h_beta.png", dpi=200)
            fig.clear()
            if h_beta_int > mean + sigma:
                print(f"H beta: {h_beta_pr} (z = {(h_beta_pr - h_beta) / h_beta})")
                out.write(f"H beta: {h_beta_pr} (z = {(h_beta_pr - h_beta) / h_beta})\n")
                z.append((h_beta_pr - h_beta) / h_beta)
                found_h_beta = True
            else:
                print("H beta: too low intensity!")
                out.write("H beta: too low intensity!\n")
        except RuntimeError:
            print("H beta: ???")
            out.write("H beta: ???\n")
    
        center = round((h_gamma - spectra[i][0][0]) / d_lambda)
        line_img = numpy.array(spectra[i][1][(center - 5):(center + 5)])
        try:
            params, gaussian_model = make_model(line_img, make_gaussian, [5, 1, spectra[i][1][5]])
            h_gamma_pr = (center - 5 + params[0]) * d_lambda + spectra[i][0][0]
            h_gamma_int = params[2]
            plt.plot(spectra[i][0][(center - 5):(center + 5)], line_img)
            plt.plot(spectra[i][0][(center - 5):(center + 5)], gaussian_model, color="#FF0000")
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.savefig(f"../spectra/tmp/{os.path.basename(file_names[i]).split('.')[0]}_h_gamma.png", dpi=200)
            fig.clear()
            if h_gamma_int > mean + sigma:
                print(f"H gamma: {h_gamma_pr} (z = {(h_gamma_pr - h_gamma) / h_gamma})")
                out.write(f"H gamma: {h_gamma_pr} (z = {(h_gamma_pr - h_gamma) / h_gamma})\n")
                z.append((h_gamma_pr - h_gamma) / h_gamma)
                found_h_gamma = True
            else:
                print("H gamma: too low intensity!")
                out.write("H gamma: too low intensity!\n")
        except RuntimeError:
            print("H gamma: ???")
            out.write("H gamma: ???\n")
        if found_h_alpha and found_h_beta:
            print(f"H alpha / H beta = {h_alpha_int / h_beta_int}")
            out.write(f"H alpha / H beta = {h_alpha_int / h_beta_int}\n")
        if found_h_beta and found_h_gamma:
            print(f"H beta / H gamma = {h_beta_int / h_gamma_int}")
            out.write(f"H beta / H gamma = {h_beta_int / h_gamma_int}\n")
        print("\n")
        out.write("\n")
    
        center = round((ca_v - spectra[i][0][0]) / d_lambda)
        line_img = numpy.array(spectra[i][1][(center - 5):(center + 5)])
        try:
            params, gaussian_model = make_model(line_img, make_gaussian, [5, 1, spectra[i][1][5]])
            ca_v_pr = (center - 5 + params[0]) * d_lambda + spectra[i][0][0]
            ca_v_int = params[2]
            plt.plot(spectra[i][0][(center - 5):(center + 5)], line_img)
            plt.plot(spectra[i][0][(center - 5):(center + 5)], gaussian_model, color="#FF0000")
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.savefig(f"../spectra/tmp/{os.path.basename(file_names[i]).split('.')[0]}_ca_v.png", dpi=200)
            fig.clear()
            if h_gamma_int > mean + sigma:
                print(f"Ca V: {ca_v_pr} (z = {(ca_v_pr - ca_v) / ca_v})")
                out.write(f"Ca V: {ca_v_pr} (z = {(ca_v_pr - ca_v) / ca_v})\n")
                # z.append((ca_v_pr - ca_v) / ca_v)
            else:
                print("Ca V: too low intensity!")
                out.write("Ca V: too low intensity!\n")
        except RuntimeError:
            print("Ca V: ???")
            out.write("Ca V: ???\n")
    
        center = round((ar_v_1 - spectra[i][0][0]) / d_lambda)
        line_img = numpy.array(spectra[i][1][(center - 5):(center + 5)])
        try:
            params, gaussian_model = make_model(line_img, make_gaussian, [5, 1, spectra[i][1][5]])
            ar_v_1_pr = (center - 5 + params[0]) * d_lambda + spectra[i][0][0]
            ar_v_1_int = params[2]
            plt.plot(spectra[i][0][(center - 5):(center + 5)], line_img)
            plt.plot(spectra[i][0][(center - 5):(center + 5)], gaussian_model, color="#FF0000")
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.savefig(f"../spectra/tmp/{os.path.basename(file_names[i]).split('.')[0]}_ar_v_1.png", dpi=200)
            fig.clear()
            if ar_v_1_int > mean + sigma:
                print(f"Ar V: {ar_v_1_pr} (z = {(ar_v_1_pr - ar_v_1) / ar_v_1})")
                out.write(f"Ar V: {ar_v_1_pr} (z = {(ar_v_1_pr - ar_v_1) / ar_v_1})\n")
                # z.append((ar_v_1_pr - ar_v_1) / ar_v_1)
            else:
                print("Ar V: too low intensity!")
                out.write("Ar V: too low intensity!\n")
        except RuntimeError:
            print("Ar V: ???")
            out.write("Ar V: ???\n")
    
        center = round((ar_v_2 - spectra[i][0][0]) / d_lambda)
        line_img = numpy.array(spectra[i][1][(center - 5):(center + 5)])
        try:
            params, gaussian_model = make_model(line_img, make_gaussian, [5, 1, spectra[i][1][5]])
            ar_v_2_pr = (center - 5 + params[0]) * d_lambda + spectra[i][0][0]
            ar_v_2_int = params[2]
            plt.plot(spectra[i][0][(center - 5):(center + 5)], line_img)
            plt.plot(spectra[i][0][(center - 5):(center + 5)], gaussian_model, color="#FF0000")
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.savefig(f"../spectra/tmp/{os.path.basename(file_names[i]).split('.')[0]}_ar_v_2.png", dpi=200)
            fig.clear()
            if ar_v_2_int > mean + sigma:
                print(f"Ar V: {ar_v_2_pr} (z = {(ar_v_2_pr - ar_v_2) / ar_v_2})")
                out.write(f"Ar V: {ar_v_2_pr} (z = {(ar_v_2_pr - ar_v_2) / ar_v_2})\n")
                # z.append((ar_v_2_pr - ar_v_2) / ar_v_2)
            else:
                print("Ar V: too low intensity!")
                out.write("Ar V: too low intensity!\n")
        except RuntimeError:
            print("Ar V: ???")
            out.write("Ar V: ???\n")
    
        center = round((k_iv - spectra[i][0][0]) / d_lambda)
        line_img = numpy.array(spectra[i][1][(center - 5):(center + 5)])
        try:
            params, gaussian_model = make_model(line_img, make_gaussian, [5, 1, spectra[i][1][5]])
            k_iv_pr = (center - 5 + params[0]) * d_lambda + spectra[i][0][0]
            k_iv_int = params[2]
            plt.plot(spectra[i][0][(center - 5):(center + 5)], line_img)
            plt.plot(spectra[i][0][(center - 5):(center + 5)], gaussian_model, color="#FF0000")
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.savefig(f"../spectra/tmp/{os.path.basename(file_names[i]).split('.')[0]}_k_iv.png", dpi=200)
            fig.clear()
            if k_iv_int > mean + sigma:
                print(f"K IV: {k_iv_pr} (z = {(k_iv_pr - k_iv) / k_iv})")
                out.write(f"K IV: {k_iv_pr} (z = {(k_iv_pr - k_iv) / k_iv})\n")
                # z.append((k_iv_pr - k_iv) / k_iv)
            else:
                print("K IV: too low intensity!")
                out.write("K IV: too low intensity!\n")
        except RuntimeError:
            print("K IV: ???")
            out.write("K IV: ???\n")
    
        center = round((cl_iv_1 - spectra[i][0][0]) / d_lambda)
        line_img = numpy.array(spectra[i][1][(center - 5):(center + 5)])
        try:
            params, gaussian_model = make_model(line_img, make_gaussian, [5, 1, spectra[i][1][5]])
            cl_iv_1_pr = (center - 5 + params[0]) * d_lambda + spectra[i][0][0]
            cl_iv_1_int = params[2]
            plt.plot(spectra[i][0][(center - 5):(center + 5)], line_img)
            plt.plot(spectra[i][0][(center - 5):(center + 5)], gaussian_model, color="#FF0000")
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.savefig(f"../spectra/tmp/{os.path.basename(file_names[i]).split('.')[0]}_cl_iv_1.png", dpi=200)
            fig.clear()
            if cl_iv_1_int > mean + sigma:
                print(f"Cl IV: {cl_iv_1_pr} (z = {(cl_iv_1_pr - cl_iv_1) / cl_iv_1})")
                out.write(f"Cl IV: {cl_iv_1_pr} (z = {(cl_iv_1_pr - cl_iv_1) / cl_iv_1})\n")
                # z.append((cl_iv_1_pr - cl_iv_1) / cl_iv_1)
            else:
                print("Cl IV: too low intensity!")
                out.write("Cl IV: too low intensity!\n")
        except RuntimeError:
            print("Cl IV: ???")
            out.write("Cl IV: ???\n")
    
        center = round((cl_iv_2 - spectra[i][0][0]) / d_lambda)
        line_img = numpy.array(spectra[i][1][(center - 5):(center + 5)])
        try:
            params, gaussian_model = make_model(line_img, make_gaussian, [5, 1, spectra[i][1][5]])
            cl_iv_2_pr = (center - 5 + params[0]) * d_lambda + spectra[i][0][0]
            cl_iv_2_int = params[2]
            plt.plot(spectra[i][0][(center - 5):(center + 5)], line_img)
            plt.plot(spectra[i][0][(center - 5):(center + 5)], gaussian_model, color="#FF0000")
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.savefig(f"../spectra/tmp/{os.path.basename(file_names[i]).split('.')[0]}_cl_iv_2.png", dpi=200)
            fig.clear()
            if cl_iv_2_int > mean + sigma:
                print(f"Cl IV: {cl_iv_2_pr} (z = {(cl_iv_2_pr - cl_iv_2) / cl_iv_2})")
                out.write(f"Cl IV: {cl_iv_2_pr} (z = {(cl_iv_2_pr - cl_iv_2) / cl_iv_2})\n")
                # z.append((cl_iv_2_pr - cl_iv_2) / cl_iv_2)
            else:
                print("Cl IV: too low intensity!")
                out.write("Cl IV: too low intensity!\n")
        except RuntimeError:
            print("Cl IV: ???")
            out.write("Cl IV: ???\n")
    
        center = round((he_ii - spectra[i][0][0]) / d_lambda)
        line_img = numpy.array(spectra[i][1][(center - 5):(center + 5)])
        try:
            params, gaussian_model = make_model(line_img, make_gaussian, [5, 1, spectra[i][1][5]])
            he_ii_pr = (center - 5 + params[0]) * d_lambda + spectra[i][0][0]
            he_ii_int = params[2]
            plt.plot(spectra[i][0][(center - 5):(center + 5)], line_img)
            plt.plot(spectra[i][0][(center - 5):(center + 5)], gaussian_model, color="#FF0000")
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.savefig(f"../spectra/tmp/{os.path.basename(file_names[i]).split('.')[0]}_he_ii.png", dpi=200)
            fig.clear()
            if he_ii_int > mean + sigma:
                print(f"He II: {he_ii_pr} (z = {(he_ii_pr - he_ii) / he_ii})\n")
                out.write(f"He II: {he_ii_pr} (z = {(he_ii_pr - he_ii) / he_ii})\n\n")
                z.append((he_ii_pr - he_ii) / he_ii)
            else:
                print("He II: too low intensity!\n")
                out.write("He II: too low intensity!\n\n")
        except RuntimeError:
            print("He II: ???\n")
            out.write("He II: ???\n\n")
    
        center = round((s_ii_1 - spectra[i][0][0]) / d_lambda)
        line_img = numpy.array(spectra[i][1][(center - 5):(center + 5)])
        try:
            params, gaussian_model = make_model(line_img, make_gaussian, [5, 1, spectra[i][1][5]])
            s_ii_1_pr = (center - 5 + params[0]) * d_lambda + spectra[i][0][0]
            s_ii_1_int = params[2]
            plt.plot(spectra[i][0][(center - 5):(center + 5)], line_img)
            plt.plot(spectra[i][0][(center - 5):(center + 5)], gaussian_model, color="#FF0000")
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.savefig(f"../spectra/tmp/{os.path.basename(file_names[i]).split('.')[0]}_s_ii_1.png", dpi=200)
            fig.clear()
            if s_ii_1_int > mean + 0.5 * sigma:
                print(f"S II: {s_ii_1_pr} (z = {(s_ii_1_pr - s_ii_1) / s_ii_1})")
                out.write(f"S II: {s_ii_1_pr} (z = {(s_ii_1_pr - s_ii_1) / s_ii_1})\n")
                z.append((s_ii_1_pr - s_ii_1) / s_ii_1)
                found_s_ii_1 = True
            else:
                print("S II: too low intensity!")
                out.write("S II: too low intensity!\n")
        except RuntimeError:
            print("S II: ???")
            out.write("S II: ???\n")
    
        center = round((s_ii_2 - spectra[i][0][0]) / d_lambda)
        line_img = numpy.array(spectra[i][1][(center - 5):(center + 5)])
        try:
            params, gaussian_model = make_model(line_img, make_gaussian, [5, 1, spectra[i][1][5]])
            s_ii_2_pr = (center - 5 + params[0]) * d_lambda + spectra[i][0][0]
            s_ii_2_int = params[2]
            plt.plot(spectra[i][0][(center - 5):(center + 5)], line_img)
            plt.plot(spectra[i][0][(center - 5):(center + 5)], gaussian_model, color="#FF0000")
            fig = plt.gcf()
            fig.set_size_inches(12, 6)
            fig.savefig(f"../spectra/tmp/{os.path.basename(file_names[i]).split('.')[0]}_s_ii_2.png", dpi=200)
            fig.clear()
            if s_ii_2_int > mean + 0.5 * sigma:
                print(f"S II: {s_ii_2_pr} (z = {(s_ii_2_pr - s_ii_2) / s_ii_2})")
                out.write(f"S II: {s_ii_2_pr} (z = {(s_ii_2_pr - s_ii_2) / s_ii_2})\n")
                z.append((s_ii_2_pr - s_ii_2) / s_ii_2)
                found_s_ii_2 = True
            else:
                print("S II: too low intensity!")
                out.write("S II: too low intensity!\n")
        except RuntimeError:
            print("S II: ???")
            out.write("S II: ???\n")
        if found_s_ii_1 and found_s_ii_2:
            print(f"[S II] = {s_ii_1_int / s_ii_2_int}")
            out.write(f"[S II] = {s_ii_1_int / s_ii_2_int}\n")
        print("\n")
        out.write("\n")
    
        print(f"z = {numpy.mean(z)} (std = {numpy.std(z)})")
        out.write(f"z = {numpy.mean(z)} (std = {numpy.std(z)})\n")
        c = 2998000
        print(f"v = {c * numpy.mean(z)} km/s (std = {numpy.std(z) * c} km/s)\n\n")
        out.write(f"v = {c * numpy.mean(z)} km/s (std = {numpy.std(z) * c} km/s)\n")
