

#define _CRT_SECURE_NO_WARNINGS
#include "raylib.h"
#include "rlgl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#pragma pack(push, 1)
struct StudioHdr
{
    int   id, version;
    char  name[64];
    int   length;
    float eyeposition[3], min[3], max[3], bbmin[3], bbmax[3];
    int   flags;
    int   numbones, boneindex;
    int   numbonecontrollers, bonecontrollerindex;
    int   numhitboxes, hitboxindex;
    int   numseq, seqindex;
    int   numseqgroups, seqgroupindex;
    int   numtextures, textureindex, texturedataindex;
    int   numskinref, numskinfamilies, skinindex;
    int   numbodyparts, bodypartindex;
    int   numattachments, attachmentindex;
    int   soundtable, soundindex, soundgroups, soundgroupindex;
    int   numtransitions, transitionindex;
};
struct StudioBone
{
    char  name[32];
    int   parent, flags;
    int   bonecontroller[6];
    float value[6], scale[6];
};
struct StudioSeqDesc
{
    char  label[32];
    float fps;
    int   flags, activity, actweight;
    int   numevents, eventindex;
    int   numframes;
    int   numpivots, pivotindex;
    int   motiontype, motionbone;
    float linearmovement[3];
    int   automoveposindex, automoveangleindex;
    float bbmin[3], bbmax[3];
    int   numblends, animindex;
    int   blendtype[2];
    float blendstart[2], blendend[2];
    int   blendparent, seqgroup, entrynode, exitnode, nodeflags, nextseq;
};
struct StudioSeqGroup { char label[32]; char name[64]; int unused1, unused2; };
struct StudioTexture { char name[64]; int flags, width, height, index; };
struct StudioBodypart { char name[64]; int nummodels, base, modelindex; };
struct StudioModel
{
    char  name[64];
    int   type;
    float boundingradius;
    int   nummesh, meshindex;
    int   numverts, vertinfoindex, vertindex;
    int   numnorms, norminfoindex, normindex;
    int   numgroups, groupindex;
};
struct StudioMesh { int numtris, triindex, skinref, numnorms, normindex; };
struct StudioTriVert { short vertindex, normindex, s, t; };
#pragma pack(pop)

static_assert(sizeof(StudioHdr) == 244, "studiohdr_t");
static_assert(sizeof(StudioBone) == 112, "mstudiobone_t");
static_assert(sizeof(StudioSeqDesc) == 176, "mstudioseqdesc_t");
static_assert(sizeof(StudioSeqGroup) == 104, "mstudioseqgroup_t");
static_assert(sizeof(StudioTexture) == 80, "mstudiotexture_t");
static_assert(sizeof(StudioBodypart) == 76, "mstudiobodyparts_t");
static_assert(sizeof(StudioModel) == 112, "mstudiomodel_t");
static_assert(sizeof(StudioMesh) == 20, "mstudiomesh_t");
static_assert(sizeof(StudioTriVert) == 8, "mstudiotrivert_t");

static const int STUDIO_NF_MASKED = 0x0040;
static const int STUDIO_LOOPING = 0x0001;
static const int MAX_VERTS = 2048;  
static const int MAX_TEX_SIZE = 512;

struct Buf
{
    std::vector<uint8_t> d;
    size_t Align() { while (d.size() % 4) d.push_back(0); return d.size(); }
    size_t PutRaw(const void* p, size_t n)
    {
        size_t o = d.size();
        if (n) d.insert(d.end(), (const uint8_t*)p, (const uint8_t*)p + n);
        return o;
    }
    template <class T> size_t Put(const T& v) { return PutRaw(&v, sizeof(T)); }
};

struct OutTex
{
    int                  w = 0, h = 0;
    bool                 masked = false;
    std::vector<uint8_t> pixels;      
    uint8_t              pal[768] = {};
};

static void Quantize(const unsigned char* px, int w, int h, int maxColors, bool masked,
    std::vector<uint8_t>& out, uint8_t pal[768])
{
    struct Bin { int n = 0; double r = 0, g = 0, b = 0; };
    std::vector<Bin> bins(32768);

    auto keyOf = [](const unsigned char* p) { return ((p[0] >> 3) << 10) | ((p[1] >> 3) << 5) | (p[2] >> 3); };
    auto chan = [](int key, int axis) { return axis == 0 ? (key >> 10) : axis == 1 ? ((key >> 5) & 31) : (key & 31); };

    for (int i = 0; i < w * h; i++)
    {
        const unsigned char* p = px + i * 4;
        if (masked && p[3] < 128) continue;
        Bin& b = bins[keyOf(p)];
        b.n++; b.r += p[0]; b.g += p[1]; b.b += p[2];
    }

    std::vector<int> all;
    for (int k = 0; k < 32768; k++) if (bins[k].n) all.push_back(k);

    std::vector<std::vector<int>> boxes;
    if (!all.empty()) boxes.push_back(all);

    while ((int)boxes.size() < maxColors)
    {
        int best = -1, bestAxis = 0, bestRange = 0;
        for (size_t i = 0; i < boxes.size(); i++)
        {
            if (boxes[i].size() < 2) continue;
            for (int a = 0; a < 3; a++)
            {
                int lo = 31, hi = 0;
                for (int k : boxes[i]) { int c = chan(k, a); lo = std::min(lo, c); hi = std::max(hi, c); }
                if (hi - lo > bestRange) { bestRange = hi - lo; best = (int)i; bestAxis = a; }
            }
        }
        if (best < 0) break;

        std::vector<int> box = std::move(boxes[best]);
        std::sort(box.begin(), box.end(), [&](int a, int b) { return chan(a, bestAxis) < chan(b, bestAxis); });

        long total = 0;
        for (int k : box) total += bins[k].n;
        long   acc = 0;
        size_t split = 1;
        for (size_t i = 0; i < box.size(); i++)
        {
            acc += bins[box[i]].n;
            if (acc * 2 >= total) { split = i + 1; break; }
        }
        if (split >= box.size()) split = box.size() - 1;

        boxes[best].assign(box.begin(), box.begin() + split);
        boxes.emplace_back(box.begin() + split, box.end());
    }

    memset(pal, 0, 768);
    int nPal = (int)boxes.size();
    for (int i = 0; i < nPal; i++)
    {
        double n = 0, r = 0, g = 0, b = 0;
        for (int k : boxes[i]) { n += bins[k].n; r += bins[k].r; g += bins[k].g; b += bins[k].b; }
        pal[i * 3 + 0] = (uint8_t)(r / n + 0.5);
        pal[i * 3 + 1] = (uint8_t)(g / n + 0.5);
        pal[i * 3 + 2] = (uint8_t)(b / n + 0.5);
    }
    if (masked) { pal[255 * 3 + 0] = 0; pal[255 * 3 + 1] = 0; pal[255 * 3 + 2] = 255; } // azul = transparente

    std::vector<int> lut(32768, -1);
    out.resize((size_t)w * h);
    for (int i = 0; i < w * h; i++)
    {
        const unsigned char* p = px + i * 4;
        if (masked && p[3] < 128) { out[i] = 255; continue; }
        int key = keyOf(p);
        if (lut[key] < 0)
        {
            const Bin& b = bins[key];
            double r = b.r / b.n, g = b.g / b.n, bl = b.b / b.n;
            double bestD = 1e18;
            int    bestI = 0;
            for (int c = 0; c < nPal; c++)
            {
                double dr = r - pal[c * 3], dg = g - pal[c * 3 + 1], db = bl - pal[c * 3 + 2];
                double dist = dr * dr + dg * dg + db * db;
                if (dist < bestD) { bestD = dist; bestI = c; }
            }
            lut[key] = bestI;
        }
        out[i] = (uint8_t)lut[key];
    }
}

static int PickSize(int v)
{
    int p = 16;
    while (p * 3 / 2 < v && p < MAX_TEX_SIZE) p <<= 1;
    return p;
}

static OutTex MakeTexture(const Material& mat)
{
    OutTex t;
    const MaterialMap& map = mat.maps[MATERIAL_MAP_ALBEDO];

    Image img = {};
    if (map.texture.id > 0 && map.texture.id != rlGetTextureIdDefault())
        img = LoadImageFromTexture(map.texture);
    if (!img.data)
        img = GenImageColor(16, 16, WHITE);

    ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    ImageColorTint(&img, map.color);

    int w = PickSize(img.width), h = PickSize(img.height);
    if (w != img.width || h != img.height) ImageResize(&img, w, h);

    const unsigned char* px = (const unsigned char*)img.data;
    for (int i = 0; i < w * h; i++)
        if (px[i * 4 + 3] < 128) { t.masked = true; break; }

    t.w = w;
    t.h = h;
    Quantize(px, w, h, t.masked ? 255 : 256, t.masked, t.pixels, t.pal);
    UnloadImage(img);
    return t;
}

struct OutMesh
{
    std::vector<float>         verts;  
    std::vector<float>         norms;
    std::vector<StudioTriVert> tris;    
    int                        skinref = 0;
};

static void ToGS(float x, float y, float z, float s, float out[3])
{
    out[0] = z * s;
    out[1] = x * s;
    out[2] = y * s;
}

static OutMesh ConvertMesh(const Mesh& me, int skinref, const OutTex& tex, float scale, bool flip)
{
    OutMesh o;
    o.skinref = skinref;

    std::map<std::array<int, 3>, int> vmap, nmap;

    auto addV = [&](const float p[3]) -> int {
        std::array<int, 3> k = { (int)lroundf(p[0] * 1000), (int)lroundf(p[1] * 1000), (int)lroundf(p[2] * 1000) };
        auto it = vmap.find(k);
        if (it != vmap.end()) return it->second;
        int idx = (int)o.verts.size() / 3;
        o.verts.insert(o.verts.end(), p, p + 3);
        vmap[k] = idx;
        return idx;
        };
    auto addN = [&](const float n[3]) -> int {
        std::array<int, 3> k = { (int)lroundf(n[0] * 1000), (int)lroundf(n[1] * 1000), (int)lroundf(n[2] * 1000) };
        auto it = nmap.find(k);
        if (it != nmap.end()) return it->second;
        int idx = (int)o.norms.size() / 3;
        o.norms.insert(o.norms.end(), n, n + 3);
        nmap[k] = idx;
        return idx;
        };

    for (int t = 0; t < me.triangleCount; t++)
    {
        int   id[3];
        float pos[3][3];
        for (int k = 0; k < 3; k++)
        {
            id[k] = me.indices ? (int)me.indices[t * 3 + k] : t * 3 + k;
            ToGS(me.vertices[id[k] * 3], me.vertices[id[k] * 3 + 1], me.vertices[id[k] * 3 + 2], scale, pos[k]);
        }

        float fn[3] = { 0, 0, 1 };
        {
            float a[3] = { pos[1][0] - pos[0][0], pos[1][1] - pos[0][1], pos[1][2] - pos[0][2] };
            float b[3] = { pos[2][0] - pos[0][0], pos[2][1] - pos[0][1], pos[2][2] - pos[0][2] };
            float c[3] = { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
            float l = sqrtf(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
            if (l > 1e-12f) { fn[0] = c[0] / l; fn[1] = c[1] / l; fn[2] = c[2] / l; }
        }

        const int order[3] = { 0, flip ? 2 : 1, flip ? 1 : 2 };
        for (int oi = 0; oi < 3; oi++)
        {
            int k = order[oi];

            float n[3] = { fn[0], fn[1], fn[2] };
            if (me.normals)
                ToGS(me.normals[id[k] * 3], me.normals[id[k] * 3 + 1], me.normals[id[k] * 3 + 2], 1.0f, n);

            float u = me.texcoords ? me.texcoords[id[k] * 2] : 0.0f;
            float v = me.texcoords ? me.texcoords[id[k] * 2 + 1] : 0.0f;

            StudioTriVert tv;
            tv.vertindex = (short)addV(pos[k]);
            tv.normindex = (short)addN(n);
            tv.s = (short)lroundf(u * tex.w);
            tv.t = (short)lroundf(v * tex.h);
            o.tris.push_back(tv);
        }
    }
    return o;
}

int main(int argc, char** argv)
{
    std::string in, out;
    float       scale = 40.0f;
    bool        flip = true;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--scale") == 0 && i + 1 < argc) scale = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--no-flip") == 0)           flip = false;
        else if (in.empty())                                  in = argv[i];
        else if (out.empty())                                 out = argv[i];
    }
    if (in.empty())
    {
        printf("Indicate your glb file: ");
        std::getline(std::cin, in);

        while (!in.empty() && (in.back() == ' ' || in.back() == '"' || in.back() == '\r')) in.pop_back();
        while (!in.empty() && (in.front() == ' ' || in.front() == '"')) in.erase(in.begin());

        if (in.empty())
        {
            printf("No indicated file.\n");
            return 1;
        }
    }
    if (out.empty())
    {
        out = in;
        size_t dot = out.find_last_of('.');
        if (dot != std::string::npos) out.erase(dot);
        out += ".mdl";
    }

    SetTraceLogLevel(LOG_WARNING);
    SetConfigFlags(FLAG_WINDOW_HIDDEN);
    InitWindow(16, 16, "glb2mdl");

    Model model = LoadModel(in.c_str());
    if (model.meshCount == 0)
    {
        printf("Error: Couldn't load '%s' (maybe it has no meshes).\n", in.c_str());
        CloseWindow();
        return 2;
    }

    std::vector<OutTex> texs;
    for (int i = 0; i < model.materialCount; i++) texs.push_back(MakeTexture(model.materials[i]));
    if (texs.size() > 100) printf("Warning: %zu texturas (engine limit: 100).\n", texs.size());

    std::vector<OutMesh> meshes;
    for (int i = 0; i < model.meshCount; i++)
    {
        int mat = model.meshMaterial[i];
        if (mat < 0 || mat >= (int)texs.size()) mat = 0;
        meshes.push_back(ConvertMesh(model.meshes[i], mat, texs[mat], scale, flip));
    }

    std::vector<std::vector<int>> groups;
    {
        int curV = 0, curN = 0;
        for (int i = 0; i < (int)meshes.size(); i++)
        {
            int nv = (int)meshes[i].verts.size() / 3, nn = (int)meshes[i].norms.size() / 3;
            if (nv > MAX_VERTS || nn > MAX_VERTS)
                printf("Warning: mesh %d has %d vertices / %d normals (limite do motor: %d). Simplify it.\n", i, nv, nn, MAX_VERTS);
            if (groups.empty() || curV + nv > MAX_VERTS || curN + nn > MAX_VERTS)
            {
                groups.emplace_back();
                curV = curN = 0;
            }
            groups.back().push_back(i);
            curV += nv;
            curN += nn;
        }
    }
    if (groups.size() > 32) printf("Warning: %zu bodyparts (limite do motor: 32).\n", groups.size());

    float bmin[3] = { 1e30f, 1e30f, 1e30f }, bmax[3] = { -1e30f, -1e30f, -1e30f };
    for (auto& m : meshes)
        for (size_t i = 0; i + 2 < m.verts.size(); i += 3)
            for (int a = 0; a < 3; a++)
            {
                bmin[a] = std::min(bmin[a], m.verts[i + a]);
                bmax[a] = std::max(bmax[a], m.verts[i + a]);
            }

    Buf b;
    b.d.assign(sizeof(StudioHdr), 0);

    StudioBone bone = {};
    strcpy(bone.name, "root");
    bone.parent = -1;
    for (int k = 0; k < 6; k++) { bone.bonecontroller[k] = -1; bone.scale[k] = 1.0f; }
    size_t boneOff = b.Align();
    b.Put(bone);

    StudioSeqGroup sg = {};
    strcpy(sg.label, "default");
    size_t sgOff = b.Align();
    b.Put(sg);

    uint16_t animZero[6] = { 0, 0, 0, 0, 0, 0 };   // offsets a 0 -> usa os valores por omissao do osso
    size_t   animOff = b.Align();
    b.PutRaw(animZero, sizeof(animZero));

    StudioSeqDesc sq = {};
    strcpy(sq.label, "idle");
    sq.fps = 30.0f;
    sq.flags = STUDIO_LOOPING;
    sq.actweight = 1;
    sq.numframes = 1;
    sq.numblends = 1;
    sq.animindex = (int)animOff;
    sq.blendend[0] = 1.0f;
    memcpy(sq.bbmin, bmin, sizeof(bmin));
    memcpy(sq.bbmax, bmax, sizeof(bmax));
    size_t seqOff = b.Align();
    b.Put(sq);

    std::vector<StudioTexture> st(texs.size());
    for (size_t i = 0; i < texs.size(); i++)
    {
        st[i] = {};
        snprintf(st[i].name, sizeof(st[i].name), "mat%zu.bmp", i);
        st[i].flags = texs[i].masked ? STUDIO_NF_MASKED : 0;
        st[i].width = texs[i].w;
        st[i].height = texs[i].h;
        st[i].index = (int)b.Align();
        b.PutRaw(texs[i].pixels.data(), texs[i].pixels.size());
        b.PutRaw(texs[i].pal, 768);
    }
    size_t texDataOff = (size_t)st[0].index;
    size_t texOff = b.Align();
    for (auto& t : st) b.Put(t);

    size_t skinOff = b.Align();
    for (size_t i = 0; i < texs.size(); i++) b.Put((short)i);

    std::vector<StudioBodypart> bodyparts;
    for (size_t gi = 0; gi < groups.size(); gi++)
    {
        std::vector<float> verts, norms;
        std::vector<StudioMesh> sm;
        std::vector<Buf>        cmds;
        std::vector<int>        normOffsets;
        float                   radius = 0;

        for (int mi : groups[gi])
        {
            const OutMesh& m = meshes[mi];
            int voff = (int)verts.size() / 3, noff = (int)norms.size() / 3;

            StudioMesh s = {};
            s.numtris = (int)m.tris.size() / 3;
            s.skinref = m.skinref;
            s.numnorms = (int)m.norms.size() / 3;
            sm.push_back(s);
            normOffsets.push_back(noff);

            Buf c;
            for (size_t t = 0; t + 2 < m.tris.size(); t += 3)
            {
                c.Put((short)3);  
                for (int k = 0; k < 3; k++)
                {
                    StudioTriVert tv = m.tris[t + k];
                    tv.vertindex = (short)(tv.vertindex + voff);
                    tv.normindex = (short)(tv.normindex + noff);
                    c.Put(tv);
                }
            }
            c.Put((short)0);     
            cmds.push_back(std::move(c));

            verts.insert(verts.end(), m.verts.begin(), m.verts.end());
            norms.insert(norms.end(), m.norms.begin(), m.norms.end());
        }

        for (size_t i = 0; i + 2 < verts.size(); i += 3)
            radius = std::max(radius, sqrtf(verts[i] * verts[i] + verts[i + 1] * verts[i + 1] + verts[i + 2] * verts[i + 2]));

        size_t nv = verts.size() / 3, nn = norms.size() / 3;
        std::vector<uint8_t> z(std::max(nv, nn), 0);   

        size_t viOff = b.Align(); b.PutRaw(z.data(), nv);
        size_t vOff = b.Align(); b.PutRaw(verts.data(), nv * 12);
        size_t niOff = b.Align(); b.PutRaw(z.data(), nn);
        size_t nOff = b.Align(); b.PutRaw(norms.data(), nn * 12);

        for (size_t i = 0; i < sm.size(); i++)
        {
            sm[i].triindex = (int)b.Align();
            b.PutRaw(cmds[i].d.data(), cmds[i].d.size());
            sm[i].normindex = (int)(nOff + (size_t)normOffsets[i] * 12);
        }

        size_t meshOff = b.Align();
        for (auto& s : sm) b.Put(s);

        StudioModel m = {};
        snprintf(m.name, sizeof(m.name), "sub%zu", gi);
        m.boundingradius = radius;
        m.nummesh = (int)sm.size();
        m.meshindex = (int)meshOff;
        m.numverts = (int)nv;
        m.vertinfoindex = (int)viOff;
        m.vertindex = (int)vOff;
        m.numnorms = (int)nn;
        m.norminfoindex = (int)niOff;
        m.normindex = (int)nOff;
        size_t modelOff = b.Align();
        b.Put(m);

        StudioBodypart bp = {};
        snprintf(bp.name, sizeof(bp.name), "body%zu", gi);
        bp.nummodels = 1;
        bp.base = 1;
        bp.modelindex = (int)modelOff;
        bodyparts.push_back(bp);
    }

    size_t bpOff = b.Align();
    for (auto& bp : bodyparts) b.Put(bp);
    b.Align();

    std::string base = out;
    size_t slash = base.find_last_of("/\\");
    if (slash != std::string::npos) base.erase(0, slash + 1);

    StudioHdr h = {};
    memcpy(&h.id, "IDST", 4);
    h.version = 10;
    strncpy(h.name, base.c_str(), sizeof(h.name) - 1);
    h.length = (int)b.d.size();
    h.eyeposition[2] = bmax[2];
    memcpy(h.min, bmin, sizeof(bmin));
    memcpy(h.max, bmax, sizeof(bmax));
    memcpy(h.bbmin, bmin, sizeof(bmin));
    memcpy(h.bbmax, bmax, sizeof(bmax));
    h.numbones = 1;  h.boneindex = (int)boneOff;
    h.numseq = 1;  h.seqindex = (int)seqOff;
    h.numseqgroups = 1;  h.seqgroupindex = (int)sgOff;
    h.numtextures = (int)texs.size();
    h.textureindex = (int)texOff;
    h.texturedataindex = (int)texDataOff;
    h.numskinref = (int)texs.size();
    h.numskinfamilies = 1;
    h.skinindex = (int)skinOff;
    h.numbodyparts = (int)bodyparts.size();
    h.bodypartindex = (int)bpOff;
    memcpy(b.d.data(), &h, sizeof(h));

    FILE* f = fopen(out.c_str(), "wb");
    if (!f)
    {
        printf("Error: Couldn't write '%s'.\n", out.c_str());
        UnloadModel(model);
        CloseWindow();
        return 3;
    }
    fwrite(b.d.data(), 1, b.d.size(), f);
    fclose(f);

    int totalTris = 0, totalVerts = 0;
    for (auto& m : meshes) { totalTris += (int)m.tris.size() / 3; totalVerts += (int)m.verts.size() / 3; }
    printf("OK: %s (%zu bytes)\n", out.c_str(), b.d.size());
    printf("    %zu textures, %zu sub-models, %d vertices, %d triangles\n",
        texs.size(), groups.size(), totalVerts, totalTris);
    printf("    size (units): %.1f x %.1f x %.1f\n",
        bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]);
    printf("Note:on this version there is no skeleton support, later versions may alter this.\n");

    UnloadModel(model);
    CloseWindow();
    return 0;
}