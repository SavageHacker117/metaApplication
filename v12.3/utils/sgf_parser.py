
import re
from typing import List, Tuple, Dict, Any

class SGFParser:
    """
    A utility to parse SGF (Smart Game Format) files into a list of moves
    and extract basic game information.
    This is a simplified parser and may not handle all SGF complexities.
    """
    def __init__(self):
        pass

    def parse_sgf(self, sgf_string: str) -> Dict[str, Any]:
        """
        Parses an SGF string and returns a dictionary containing game info and moves.
        """
        game_info = {}
        moves = []

        # Extract properties (e.g., SZ, KM, PB, PW)
        properties = re.findall(r'([A-Z]{1,2})\[([^\]]*)\]', sgf_string)
        for key, value in properties:
            game_info[key] = value

        # Extract moves
        # Matches ;B[xy] or ;W[xy] or ;B[] (pass) or ;W[] (pass)
        move_pattern = re.compile(r';([BW])\[([a-z]{2}|)\]')
        for match in move_pattern.finditer(sgf_string):
            player = match.group(1)
            coords = match.group(2)
            
            if coords == "": # Pass move
                moves.append((None, None))
            else:
                col = ord(coords[0]) - ord("a")
                row = ord(coords[1]) - ord("a")
                moves.append((row, col))
        
        game_info["moves"] = moves
        return game_info

    def get_moves_from_sgf(self, sgf_string: str) -> List[Tuple[int, int]]:
        """
        Extracts only the list of moves from an SGF string.
        """
        parsed_data = self.parse_sgf(sgf_string)
        return parsed_data.get("moves", [])

# Example Usage:
# if __name__ == "__main__":
#     parser = SGFParser()
#     
#     # Example SGF string
#     sgf_example = "(;GM[1]FF[4]CA[UTF-8]AP[GoGui:1.4.9]SZ[9]KM[6.5]PB[Black]PW[White];B[dd];W[ff];B[gg];W[];B[])"
#     
#     parsed_data = parser.parse_sgf(sgf_example)
#     print("\n--- Parsed SGF Data ---")
#     print(parsed_data)
#
#     moves_only = parser.get_moves_from_sgf(sgf_example)
#     print("\n--- Moves Only ---")
#     print(moves_only)
#
#     sgf_example_2 = "(;GM[1]FF[4]SZ[19];B[pd];W[dp];B[fq];W[eq];B[dq];W[cp];B[cq];W[bq];B[bp];W[co];B[do];W[dn];B[en];W[em];B[fm];W[fn];B[gn];W[go];B[ho];W[hn];B[io];W[in];B[jo];W[jn];B[ko];W[kn];B[lo];W[ln];B[mo];W[mn];B[no];W[nn];B[oo];W[on];B[po];W[pn];B[qo];W[qn];B[ro];W[rn];B[so];W[sn];B[to];W[tn];B[up];W[un];B[vp];W[vn];B[wp];W[wn];B[xp];W[xn];B[yp];W[yn];B[zp];W[zn];B[aq];W[ap];B[ar];W[br];B[cr];W[dr];B[er];W[fr];B[gr];W[hr];B[ir];W[jr];B[kr];W[lr];B[mr];W[nr];B[or];W[pr];B[qr];W[rr];B[sr];W[tr];B[ur];W[vr];B[wr];W[xr];B[yr];W[zr];B[as];W[bs];B[cs];W[ds];B[es];W[fs];B[gs];W[hs];B[is];W[js];B[ks];W[ls];B[ms];W[ns];B[os];W[ps];B[qs];W[rs];B[ss];W[ts];B[us];W[vs];B[ws];W[xs];B[ys];W[zs];B[at];W[bt];B[ct];W[dt];B[et];W[ft];B[gt];W[ht];B[it];W[jt];B[kt];W[lt];B[mt];W[nt];B[ot];W[pt];B[qt];W[rt];B[st];W[tt];B[ut];W[vt];B[wt];W[xt];B[yt];W[zt];B[au];W[bu];B[cu];W[du];B[eu];W[fu];B[gu];W[hu];B[iu];W[ju];B[ku];W[lu];B[mu];W[nu];B[ou];W[pu];B[qu];W[ru];B[su];W[tu];B[uu];W[vu];B[wu];W[xu];B[yu];W[zu];B[av];W[bv];B[cv];W[dv];B[ev];W[fv];B[gv];W[hv];B[iv];W[jv];B[kv];W[lv];B[mv];W[nv];B[ov];W[pv];B[qv];W[rv];B[sv];W[tv];B[uv];W[vv];B[wv];W[xv];B[yv];W[zv];B[aw];W[bw];B[cw];W[dw];B[ew];W[fw];B[gw];W[hw];B[iw];W[jw];B[kw];W[lw];B[mw];W[nw];B[ow];W[pw];B[qw];W[rw];B[sw];W[tw];B[uw];W[vw];B[ww];W[xw];B[yw];W[zw];B[ax];W[bx];B[cx];W[dx];B[ex];W[fx];B[gx];W[hx];B[ix];W[jx];B[kx];W[lx];B[mx];W[nx];B[ox];W[px];B[qx];W[rx];B[sx];W[tx];B[ux];W[vx];B[wx];W[xx];B[yx];W[zx];B[ay];W[by];B[cy];W[dy];B[ey];W[fy];B[gy];W[hy];B[iy];W[jy];B[ky];W[ly];B[my];W[ny];B[oy];W[py];B[qy];W[ry];B[sy];W[ty];B[uy];W[vy];B[wy];W[xy];B[yy];W[zy];B[az];W[bz];B[cz];W[dz];B[ez];W[fz];B[gz];W[hz];B[iz];W[jz];B[kz];W[lz];B[mz];W[nz];B[oz];W[pz];B[qz];W[rz];B[sz];W[tz];B[uz];W[vz];B[wz];W[xz];B[yz];W[zz];B[];W[])"
#     parsed_data_2 = parser.parse_sgf(sgf_example_2)
#     print("\n--- Parsed SGF Data 2 ---")
#     print(parsed_data_2)


