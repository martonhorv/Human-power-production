import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# 3-P MODEL


# add paths
input_excel =    r""   
sheet_name = ""
output_excel = r""
residual_excel = r""


def unified_model(x, t):
    W = x[0]
    CP = x[1]
    return W/t + CP



def residuals(x, t, y):
    return unified_model(x, t) - y


def r2(y, yfit):
    ss_res = np.sum((y-yfit)**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    return 1 - ss_res/ss_tot

def aic(y, yfit, k):
    n = len(y)
    rss = np.sum((y-yfit)**2)
    return n*np.log(rss/n) + 2*k


df = pd.read_excel(input_excel, sheet_name=sheet_name)

t_all = pd.to_numeric(df.iloc[:,0],errors="coerce")

summary_rows = []
correlation_sheets = {}

#filter durations included

time_values = [1, 2, 3, 5, 7, 11, 16, 25, 37, 55, 83, 125, 190, 280, 420, 630, 960, 1440, 2160, 3240, 4800, 7200, 10800]

residual_df = {"time": time_values}

for subject in df.columns[1:]: 

    P_all = pd.to_numeric(df[subject], errors="coerce").values

    mask_t = (np.isfinite(t_all) & np.isfinite(P_all) & (t_all > 0))


    mask_t = t_all[t_all.isin(time_values)].index  

    t = t_all[mask_t]
    P_t = P_all[mask_t]

    try:
        x0 = [10000, 400]  

        res = least_squares(residuals, x0=x0, args=(t, P_t), method="trf", loss="linear", verbose=0)

        x_opt = res.x

        P_fit_t = unified_model(x_opt, t)

        residual_values = np.full(len(time_values), np.nan)

        residual_values = (P_fit_t-P_t)                    

        residual_df[subject] = residual_values


        W = x_opt[0]
        CP = x_opt[1]
        
        m = len(P_t)
        p = len(x_opt)

        sigma2 = np.sum(res.fun**2)/(m-p)

        J = res.jac

        JTJ = J.T @ J

        rank = np.linalg.matrix_rank(J)
        cond = np.linalg.cond(J)

        cov = sigma2 * np.linalg.pinv(JTJ)

        se = np.sqrt(np.diag(cov))

        corr = cov / np.outer(se, se)

        R2 = r2(P_t, P_fit_t)

        AIC = aic(P_t, P_fit_t, k=p)

        RMSE = np.sqrt(np.mean((P_t-P_fit_t)**2))
        MAE = np.mean(np.abs(P_t - P_fit_t))
        RMSRE = np.sqrt(np.mean(((P_t - P_fit_t) / P_t)**2))
        MARE = np.mean(np.abs((P_t - P_fit_t) / P_t))

        summary_rows.append({
            "Subject": subject,
            "1-s power": max(P_t),
            "W'": W,
            "CP": CP,
            "SE_W": se[0],
            "SE_CP": se[1],
            "R2": R2,
            "AIC": AIC,

            "RMSE": RMSE,
            "MAE": MAE,
            "RMSRE": RMSRE,
            "MARE": MARE,

            "Rank": rank,
            "Condition": cond,
            "Max correlation":np.max(np.abs(corr[np.triu_indices(p,1)])),
        })

        corr_df = pd.DataFrame(
            corr,
            columns=[
                "W'",
                "CP",
            ],
            index=[
                "W'",
                "CP",
            ]
        )

        correlation_sheets[subject] = corr_df

    except Exception as e:
        print(f"Failed: {subject}")
        print(e)


summary_df = pd.DataFrame(summary_rows)


residual_df = pd.DataFrame(residual_df)


with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    summary_df.to_excel(writer, index=False)

with pd.ExcelWriter(residual_excel, engine="openpyxl") as writer:
    residual_df.to_excel(writer, index=False)

   for subject, corr_df in correlation_sheets.items():
        sheet = (str(subject)[:26]+ "_corr")
        corr_df.to_excel(writer,sheet_name=sheet)

