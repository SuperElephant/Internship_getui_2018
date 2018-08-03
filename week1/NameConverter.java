/**
 * @author Chao Xiang
 * @Date 2018-07
 *
 * Fire name: NameConverter.java
 * It is hive UDF used for convert the pkg_name into app_name base on the given dic.
 *
 * Notice:
 *  Dict path is set as 'System.getProperty("user.dir")+"/raw_dict"'
 *  Only support Text (single pkg name) --> Text (array is not supported), presently.
 */


import org.apache.hadoop.io.Text;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;

import java.io.*;
import java.util.HashMap;

@Description(name = "convert_name",
        value = "_FUNC_(pkg_name) - Returns the corresponding APP name of package name,"
                +" if not no corresponding APP name found return NULL",
        extended = "Example:\n"
                + "  > SELECT GUI, convert_name(pkg_name[0]) AS app_name FROM data_source")

public class NameConverter extends UDF {

    private final String DICT_PATH = System.getProperty("user.dir")+"/raw_dict";
    // private final String DICT_PATH = "/Users/chaoxiang/IdeaProjects/testa/src/main/resources/raw_dict";
    // private final String DICT_PATH = NameConverter.class.getResource("/").toString();

    private final String SEPARATOR = "\\|";

    private HashMap<Text,Text> name_map;

    public NameConverter(){
        // import the pkg_name - app_name table

        name_map = importDict();
    }

    /**
     * Generate a dict for further use
     * dict is  read from {@link #DICT_PATH}
     * @return HashMap<Text, Text> dict
     */
    private HashMap<Text,Text> importDict(){
        String line = "";
        HashMap<Text, Text> dict = new HashMap<Text, Text>();
        String[] names = null;
        try {
            BufferedReader reader = new BufferedReader(new FileReader(DICT_PATH));
            line = reader.readLine();
            while (line != null){
                names = line.split(SEPARATOR,2);
                if (names.length!=2){
                    print("importDict: error " + line);
                }else {
                    dict.put(new Text(names[0]),new Text(names[1]));
                }
                line = reader.readLine();
            }
        } catch (FileNotFoundException e){
            print(System.getProperty("user.dir"));
            print(NameConverter.class.getResource("/").toString());
            print("file: " + DICT_PATH + " not found");
        } catch (IOException e){
            print(e.toString());
        }
        return dict;
    }

    private void print(String content){
        System.out.println("NameConverter: " + content);
    }

    /**
     * Give a corresponding app name of a certain pkg name
     * @param pkg_name
     * @return app_name
     */
    public Text evaluate(Text pkg_name){
        if (pkg_name == null){ return null; }
        Text app_name = null;
        if (name_map.containsKey(pkg_name)) {
            app_name = name_map.get(pkg_name);
        }else{
            print("No corresponding app name found: " + pkg_name.toString());
        }
        return app_name;
    }




}
